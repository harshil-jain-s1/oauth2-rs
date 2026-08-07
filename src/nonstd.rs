//! Support for non-standard OAuth2-like providers via jq-style JSON
//! transforms threaded through the standard `Client`/token-request
//! machinery, rather than a bespoke parallel client. See
//! [`NonStdCompat`] and [`crate::Client::set_nonstd_compat`].

/// Bundles the three knobs needed to talk to a non-standards-compliant
/// token endpoint:
/// - `req_map`: a jq filter applied to the outgoing request body (as a flat
///   JSON object of the same fields this crate would otherwise
///   form-encode) before it is sent. When set, the request is sent as a
///   JSON body (`Content-Type: application/json`) instead of
///   form-urlencoded.
/// - `res_map`: a jq filter applied to the raw JSON response body before
///   this crate's standard deserialization runs.
/// - `res_type`: overrides the expected response `Content-Type` (and the
///   outgoing `Accept` header), replacing this crate's built-in
///   `application/json` assumption.
///
/// Attach via [`Client::set_nonstd_compat`](crate::Client::set_nonstd_compat).
/// Actually applying `req_map`/`res_map` (via the pure-Rust `jaq` jq
/// engine) requires the `nonstd-compat` feature; the setters that populate
/// this struct's fields are gated on that feature.
#[derive(Clone, Debug, Default)]
pub struct NonStdCompat {
    pub(crate) req_map: Option<String>,
    pub(crate) res_map: Option<String>,
    pub(crate) res_type: Option<String>,
}

impl NonStdCompat {
    /// Creates an empty configuration; use the `with_*` methods to set
    /// individual fields.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the jq filter applied to the outgoing request body (a flat JSON
    /// object of the same fields this crate would otherwise form-encode,
    /// e.g. `grant_type`, `client_id`, `client_secret`, `scope`) before it
    /// is sent. Switches the request body to JSON.
    pub fn with_req_map(mut self, req_map: impl Into<String>) -> Self {
        self.req_map = Some(req_map.into());
        self
    }

    /// Sets the jq filter applied to the raw JSON response body before this
    /// crate's standard deserialization runs.
    pub fn with_res_map(mut self, res_map: impl Into<String>) -> Self {
        self.res_map = Some(res_map.into());
        self
    }

    /// Overrides the expected response `Content-Type` (and the outgoing
    /// `Accept` header), replacing the default `application/json` check.
    pub fn with_res_type(mut self, res_type: impl Into<String>) -> Self {
        self.res_type = Some(res_type.into());
        self
    }
}

/// `jaq` builtins excluded from every filter's function table. `filter_src`
/// commonly comes from operator- or even end-user-supplied configuration, so
/// it is treated as fully untrusted:
/// - `env` (`jaq-std`) exposes the whole process environment as a JSON
///   object — a filter could otherwise smuggle arbitrary env vars (cloud
///   credentials, other secrets) into the mapped request/response body.
/// - `now` exposes the process's wall-clock time; excluded defensively even
///   though it's low-severity, since a filter has no legitimate need for it.
///
/// `debug`/`stderr` (which write the current value to the host's `log`
/// facade — a narrower side channel than `env`, and only observable if
/// DEBUG-level logging is ever enabled) are deliberately **not** excluded:
/// `jaq-std`'s own prelude (`defs.jq`) unconditionally defines `debug`/
/// `stderr` in terms of the native `debug_empty`/`stderr_empty` filters, and
/// the whole prelude is compiled as one unit regardless of what a given
/// filter actually calls — removing those two natives makes *every* filter
/// fail to compile, not just ones that use them. Excluding them isn't
/// achievable without forking `jaq-std` to also drop or rewrite those two
/// prelude definitions.
///
/// This also does **not** stop `repeat`/`recurse`/`while`/`until`: those are
/// `def`-based recursive definitions baked into `jaq-core`'s own trusted
/// prelude (`defs.jq`), not native functions reachable via this table, so
/// they can't be filtered out this way without forking and hand-maintaining
/// a trimmed copy of that prelude. [`JQ_EXEC_TIMEOUT`] is what actually
/// bounds those (and any other unbounded-looping construct).
const EXCLUDED_FUNS: &[&str] = &["env", "now"];

/// Serialized jq output larger than this is rejected. Guards against a
/// filter that terminates within [`JQ_EXEC_TIMEOUT`] but produces an
/// excessively large result.
const JQ_MAX_OUTPUT_BYTES: usize = 64 * 1024;

/// Hard wall-clock budget for a single jq filter execution.
///
/// `jaq` has no built-in step, recursion-depth, or timeout budget — its
/// evaluator will run a filter like `reduce repeat(1) as $x (0; .+1)`
/// forever, using only prelude functions and core language syntax (no
/// excludable builtin, no user-defined `def`). Since [`run_jq`] executes
/// synchronously and is invoked from synchronous code called inline from
/// async token-request futures, an unbounded loop would otherwise
/// permanently block whichever async runtime worker thread happens to be
/// polling it — and since that runtime is typically shared across an
/// entire host process, this can escalate from "one login attempt hangs"
/// to "the whole process wedges." Every execution therefore runs on its
/// own dedicated thread, joined with this timeout; on timeout the orphaned
/// thread is left to finish (or loop forever) in the background rather
/// than being forcibly killed, since Rust has no safe mechanism to
/// preempt a running thread. This bounds the *visible* impact to one
/// failed request instead of an unbounded hang.
const JQ_EXEC_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(2);

/// Returns `true` if `filter_src` lexically contains a `def` (function
/// definition) anywhere, including nested inside `(...)`/`[...]`/`{...}`
/// blocks or string interpolation. Reshaping a request/response body never
/// needs a custom function; rejecting `def` outright closes the most direct
/// route to a user-authored infinite-recursion helper (e.g. `def f: f; f`)
/// with a clear error instead of a timeout. This does not (and cannot, by
/// itself) stop the prelude's own recursive functions (`repeat`/`recurse`/
/// `while`/`until`) — see [`JQ_EXEC_TIMEOUT`] for that.
///
/// Returns `false` (i.e. does not flag) on a lex error — an invalid filter
/// still fails normally at the parse step in [`run_jq_inner`], with that
/// step's own error message.
fn contains_user_def(filter_src: &str) -> bool {
    use jaq_core::load::lex::{Lexer, StrPart, Tok, Token};

    fn walk(tokens: &[Token<&str>]) -> bool {
        tokens.iter().any(|Token(s, tok)| match tok {
            Tok::Word => *s == "def",
            Tok::Block(inner) => walk(inner),
            Tok::Str(parts) => parts.iter().any(|part| match part {
                StrPart::Term(inner) => walk(std::slice::from_ref(inner)),
                _ => false,
            }),
            _ => false,
        })
    }

    Lexer::new(filter_src)
        .lex()
        .map(|tokens| walk(&tokens))
        .unwrap_or(false)
}

/// Compiles and runs a `jq` filter (via the pure-Rust `jaq` crate family)
/// against a single JSON input, returning its first output value.
///
/// `filter_src` is untrusted config (see [`EXCLUDED_FUNS`] and
/// [`JQ_EXEC_TIMEOUT`] for the specific protections this applies).
#[cfg(feature = "nonstd-compat")]
pub(crate) fn run_jq(
    filter_src: &str,
    input: serde_json::Value,
) -> Result<serde_json::Value, String> {
    if contains_user_def(filter_src) {
        return Err(format!(
            "jq filter {filter_src:?} defines a custom function (`def`), which is not allowed"
        ));
    }

    let owned_filter_src = filter_src.to_owned();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(run_jq_inner(&owned_filter_src, input));
    });
    let output = rx
        .recv_timeout(JQ_EXEC_TIMEOUT)
        .map_err(|_| format!("jq filter {filter_src:?} execution exceeded {JQ_EXEC_TIMEOUT:?} timeout"))??;

    let serialized_len = serde_json::to_vec(&output)
        .map_err(|err| err.to_string())?
        .len();
    if serialized_len > JQ_MAX_OUTPUT_BYTES {
        return Err(format!(
            "jq filter {filter_src:?} output exceeded {JQ_MAX_OUTPUT_BYTES} bytes ({serialized_len} bytes)"
        ));
    }

    Ok(output)
}

fn run_jq_inner(filter_src: &str, input: serde_json::Value) -> Result<serde_json::Value, String> {
    use jaq_core::load::{Arena, File, Loader};
    use jaq_core::{data, unwrap_valr, Compiler, Ctx, Vars};
    use jaq_json::Val;

    let defs = jaq_core::defs()
        .chain(jaq_std::defs())
        .chain(jaq_json::defs());
    let funs = jaq_core::funs()
        .chain(jaq_std::funs())
        .chain(jaq_json::funs())
        .filter(|(name, ..)| !EXCLUDED_FUNS.contains(name));

    let loader = Loader::new(defs);
    let arena = Arena::default();
    let modules = loader
        .load(
            &arena,
            File {
                code: filter_src,
                path: (),
            },
        )
        .map_err(|err| format!("failed to parse jq filter {filter_src:?}: {err:?}"))?;

    let filter = Compiler::default()
        .with_funs(funs)
        .compile(modules)
        .map_err(|err| format!("failed to compile jq filter {filter_src:?}: {err:?}"))?;

    let input: Val = serde_json::from_value(input).map_err(|err| err.to_string())?;
    let ctx = Ctx::<data::JustLut<Val>>::new(&filter.lut, Vars::new([]));

    let output = filter
        .id
        .run((ctx, input))
        .map(unwrap_valr)
        .next()
        .ok_or_else(|| format!("jq filter {filter_src:?} produced no output"))?
        .map_err(|err| format!("jq filter {filter_src:?} failed at runtime: {err:?}"))?;

    serde_json::from_str(&output.to_string()).map_err(|err| err.to_string())
}

#[cfg(all(test, feature = "nonstd-compat"))]
mod tests {
    use super::*;

    #[test]
    fn runs_a_simple_filter() {
        let output = run_jq("del(.grant_type)", serde_json::json!({
            "grant_type": "client_credentials",
            "client_id": "test",
        }))
        .unwrap();
        assert_eq!(output, serde_json::json!({"client_id": "test"}));
    }

    #[test]
    fn compile_error_is_reported() {
        let err = run_jq("this is not jq", serde_json::json!({})).unwrap_err();
        assert!(err.contains("failed to"), "unexpected error: {err}");
    }

    #[test]
    fn runtime_error_is_reported() {
        let err = run_jq(r#"error("boom")"#, serde_json::json!({})).unwrap_err();
        assert!(
            err.contains("failed at runtime"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn env_is_not_available() {
        let err = run_jq("env", serde_json::json!({})).unwrap_err();
        assert!(
            err.contains("failed to compile"),
            "expected a compile error since `env` should be undefined, got: {err}"
        );
    }

    #[test]
    fn now_is_not_available() {
        let err = run_jq("now", serde_json::json!({})).unwrap_err();
        assert!(
            err.contains("failed to compile"),
            "expected a compile error since `now` should be undefined, got: {err}"
        );
    }

    #[test]
    fn user_defined_functions_are_rejected() {
        let err = run_jq("def f: f; f", serde_json::json!({})).unwrap_err();
        assert!(
            err.contains("defines a custom function"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn user_defined_functions_nested_in_a_block_are_rejected() {
        // `def` doesn't have to appear at the top level of the filter to be
        // dangerous - it just has to be reachable by the lexer, including
        // inside a parenthesized/bracketed/braced block.
        let err = run_jq("[(def f: f; f)]", serde_json::json!({})).unwrap_err();
        assert!(
            err.contains("defines a custom function"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn unbounded_execution_times_out() {
        // Uses only prelude functions (`repeat`) and core language syntax
        // (`reduce ... as $x (...)`) - no excluded builtin, no user `def` -
        // proving layers B/C alone don't stop unbounded recursion, and that
        // the execution timeout is the thing that actually bounds it.
        let err = run_jq(
            "reduce repeat(1) as $x (0; .+1)",
            serde_json::json!({}),
        )
        .unwrap_err();
        assert!(
            err.contains("timeout"),
            "expected a timeout error, got: {err}"
        );
    }

    #[test]
    fn oversized_output_is_rejected() {
        let err = run_jq("[range(0;50000)]", serde_json::json!({})).unwrap_err();
        assert!(
            err.contains("exceeded") && err.contains("bytes"),
            "unexpected error: {err}"
        );
    }
}
