//! Support for non-standard OAuth2-like providers via jq-style JSON
//! transforms threaded through the standard `Client`/token-request
//! machinery, rather than a bespoke parallel client. See
//! [`NonStdCompat`] and [`crate::Client::set_nonstd_compat`].

use std::fmt;
use std::sync::Arc;

/// Bundles the three knobs needed to talk to a non-standards-compliant
/// token endpoint:
/// - `req_map`: a jq filter applied to the outgoing request body (as a flat
///   JSON object of the same fields this crate would otherwise
///   form-encode) before it is sent. The filter's output controls both the
///   request body and its `Content-Type` — see [`CompiledFilter`]'s module
///   docs for the exact contract.
/// - `res_map`: a jq filter applied to the raw JSON response body before
///   this crate's standard deserialization runs.
/// - `res_type`: overrides the expected response `Content-Type` (and the
///   outgoing `Accept` header), replacing this crate's built-in
///   `application/json` assumption.
///
/// `req_map`/`res_map` are compiled once, in [`NonStdCompat::build`], rather
/// than on every request/response - call `.build()` after the `with_*`
/// calls and before [`Client::set_nonstd_compat`](crate::Client::set_nonstd_compat).
/// Actually applying `req_map`/`res_map` (via the pure-Rust `jaq` jq engine)
/// requires the `nonstd-compat` feature; the setters that populate this
/// struct's fields are gated on that feature.
#[derive(Clone, Debug)]
pub struct NonStdCompat {
    req_map_src: Option<String>,
    res_map_src: Option<String>,
    pub(crate) req_map: Option<Arc<CompiledFilter>>,
    pub(crate) res_map: Option<Arc<CompiledFilter>>,
    pub(crate) res_type: Option<String>,
    denylisted_idents: Vec<String>,
}

// Not `#[derive(Default)]`: a derived Default would silently zero
// `denylisted_idents` to an empty Vec (i.e. no protection at all) for
// anyone using `NonStdCompat::new()`/`::default()` without an explicit
// `with_denylisted_idents` call. Populate it from `DEFAULT_DENYLISTED_IDENTS`
// instead, so the out-of-the-box default stays protected.
impl Default for NonStdCompat {
    fn default() -> Self {
        Self {
            req_map_src: None,
            res_map_src: None,
            req_map: None,
            res_map: None,
            res_type: None,
            denylisted_idents: DEFAULT_DENYLISTED_IDENTS
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
    }
}

impl NonStdCompat {
    /// Creates a configuration with [`DEFAULT_DENYLISTED_IDENTS`] and no
    /// filters set; use the `with_*` methods to set individual fields, then
    /// [`Self::build`] to compile the filters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the jq filter applied to the outgoing request body before it is
    /// sent; see [`CompiledFilter`]'s module docs for the exact contract.
    /// Compiled by [`Self::build`].
    pub fn with_req_map(mut self, req_map: impl Into<String>) -> Self {
        self.req_map_src = Some(req_map.into());
        self
    }

    /// Sets the jq filter applied to the raw JSON response body before this
    /// crate's standard deserialization runs. Compiled by [`Self::build`].
    ///
    /// Runs for both successful and error (non-2xx) responses, with the
    /// HTTP status code available as the `$status` variable, so the filter
    /// can branch on it, e.g. `if $status == 200 then {..} else {error:
    /// ..} end`. A filter that ignores `$status` and only handles the
    /// success shape will have that same output parsed as the error type
    /// on non-2xx responses too.
    pub fn with_res_map(mut self, res_map: impl Into<String>) -> Self {
        self.res_map_src = Some(res_map.into());
        self
    }

    /// Overrides the expected response `Content-Type` (and the outgoing
    /// `Accept` header), replacing the default `application/json` check.
    pub fn with_res_type(mut self, res_type: impl Into<String>) -> Self {
        self.res_type = Some(res_type.into());
        self
    }

    /// Overrides [`DEFAULT_DENYLISTED_IDENTS`] wholesale (not additive) — the
    /// identifiers [`Self::build`] rejects `req_map`/`res_map` for referencing
    /// anywhere. Rarely needed: removing an entry (e.g. `"recurse"`) removes
    /// that specific protection for *this* filter pair, so only do so for a
    /// specific, reviewed reason (e.g. a trusted, reviewed filter that
    /// legitimately needs a provably-bounded `recurse`). Adding entries beyond
    /// the default set is a no-op unless the filter would otherwise reference
    /// that identifier. See the default list's own docs for why this check is
    /// a heuristic, not a termination proof.
    pub fn with_denylisted_idents<I, S>(mut self, idents: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.denylisted_idents = idents.into_iter().map(Into::into).collect();
        self
    }

    /// Compiles any `req_map`/`res_map` jq filter source set via `with_*`
    /// into a [`CompiledFilter`], so it's parsed and compiled exactly once
    /// rather than on every request/response. Returns the first compile
    /// error encountered, if any.
    ///
    /// Requires the "nonstd-compat" feature. Must be called before
    /// [`Client::set_nonstd_compat`](crate::Client::set_nonstd_compat) for
    /// `req_map`/`res_map` to actually take effect.
    #[cfg(feature = "nonstd-compat")]
    pub fn build(mut self) -> Result<Self, String> {
        if let Some(src) = self.req_map_src.take() {
            self.req_map = Some(Arc::new(CompiledFilter::new(
                &src,
                &self.denylisted_idents,
                &[],
            )?));
        }
        if let Some(src) = self.res_map_src.take() {
            self.res_map = Some(Arc::new(CompiledFilter::new(
                &src,
                &self.denylisted_idents,
                &["$status"],
            )?));
        }
        Ok(self)
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
/// a trimmed copy of that prelude. [`contains_denylisted_ident`] is the
/// (heuristic, best-effort) mitigation for those instead.
const EXCLUDED_FUNS: &[&str] = &["env", "now"];

/// Default identifiers [`NonStdCompat::build`] rejects `req_map`/`res_map`
/// for referencing anywhere, checked by [`contains_denylisted_ident`].
/// Override via [`NonStdCompat::with_denylisted_idents`].
///
/// - `def`: reshaping a request/response body never needs a custom
///   function; rejecting `def` outright closes the most direct route to a
///   user-authored infinite-recursion helper (e.g. `def f: f; f`).
/// - `repeat`/`recurse`/`while`/`until`: `jaq-std`'s prelude-defined,
///   `def`-based unbounded-iteration primitives (not reachable via
///   [`EXCLUDED_FUNS`], since they aren't natives) - each of these *can*
///   loop forever depending on how it's used (e.g. `repeat(f)` with no
///   `limit(...)` around it, or a `while`/`until` condition that never
///   flips), even without writing a custom `def` at all.
/// - `infinite`: the usual building block for an unbounded `range` (e.g.
///   `range(0; infinite)`), the other common no-`def` way to construct an
///   endless generator.
///
/// This is a **heuristic denylist, not a termination proof**: there's no
/// way to statically prove an arbitrary jq filter terminates, and this
/// list only covers jq's *named* unbounded-iteration constructs - it
/// can't catch every possible way to write one (e.g. a degenerate
/// `range(0; 1; 0)` with a zero step doesn't reference any denylisted
/// identifier). It also over-rejects: every use of `repeat`/`recurse`/
/// `while`/`until`/`infinite` is rejected, including ones that provably
/// terminate (e.g. `while(. < 10; . + 1)`, or bare `recurse` over
/// ordinary acyclic JSON) and even non-executable uses like an object key
/// literally named `repeat` (`{repeat: 1}`) - this check can't distinguish
/// those from a genuine infinite loop, since it's purely lexical, not an
/// analysis of what the filter actually does. Filters are expected to be
/// trusted, reviewed configuration rather than raw end-user input; this
/// check is defense in depth on top of that, not a substitute for it.
pub const DEFAULT_DENYLISTED_IDENTS: &[&str] =
    &["def", "repeat", "recurse", "while", "until", "infinite"];

/// Returns `true` if `filter_src` lexically contains any of
/// `denylisted_idents` anywhere, including nested inside
/// `(...)`/`[...]`/`{...}` blocks or string interpolation.
///
/// Returns `false` (i.e. does not flag) on a lex error — an invalid filter
/// still fails normally at the parse step in [`CompiledFilter::new`], with
/// that step's own error message.
#[cfg(feature = "nonstd-compat")]
fn contains_denylisted_ident(filter_src: &str, denylisted_idents: &[String]) -> bool {
    use jaq_core::load::lex::{Lexer, StrPart, Tok, Token};

    fn walk(tokens: &[Token<&str>], denylisted_idents: &[String]) -> bool {
        tokens.iter().any(|Token(s, tok)| match tok {
            Tok::Word => denylisted_idents.iter().any(|d| d == s),
            Tok::Block(inner) => walk(inner, denylisted_idents),
            Tok::Str(parts) => parts.iter().any(|part| match part {
                StrPart::Term(inner) => walk(std::slice::from_ref(inner), denylisted_idents),
                _ => false,
            }),
            _ => false,
        })
    }

    Lexer::new(filter_src)
        .lex()
        .map(|tokens| walk(&tokens, denylisted_idents))
        .unwrap_or(false)
}

/// A `jq` filter (run via the pure-Rust `jaq` crate family) parsed and
/// compiled exactly once via [`CompiledFilter::new`], then run against as
/// many JSON inputs as needed via [`CompiledFilter::run`].
///
/// `req_map`'s filter is expected to produce `{"content_type": ...,
/// "body": ...}`: `content_type` selects how `body` is encoded into the
/// outgoing request - `application/json` and `application/x-www-form-urlencoded`
/// are both understood directly, and any other `content_type` is supported
/// too, provided `body` is given as a base64-encoded string of the raw
/// bytes to send verbatim - letting a filter reshape the request while
/// choosing whichever encoding the target endpoint expects. `res_map`'s
/// filter is expected to produce the response body shape this crate's
/// standard token/introspection response types deserialize from; if the
/// response isn't JSON, it's handed to the filter as a base64-encoded
/// string instead of raw bytes.
pub(crate) struct CompiledFilter {
    /// Kept only for error messages.
    src: String,
    #[cfg(feature = "nonstd-compat")]
    filter: jaq_core::Filter<jaq_core::data::JustLut<jaq_json::Val>>,
}

impl fmt::Debug for CompiledFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledFilter")
            .field("src", &self.src)
            .finish()
    }
}

impl CompiledFilter {
    /// Compiles a jq filter once (the expensive part: parses the whole
    /// prelude alongside `filter_src`), rejecting it outright if it
    /// references any of `denylisted_idents`
    /// (see [`NonStdCompat::with_denylisted_idents`]).
    #[cfg(feature = "nonstd-compat")]
    pub(crate) fn new(
        filter_src: &str,
        denylisted_idents: &[String],
        global_vars: &[&str],
    ) -> Result<Self, String> {
        use jaq_core::load::{Arena, File, Loader};
        use jaq_core::Compiler;

        if contains_denylisted_ident(filter_src, denylisted_idents) {
            return Err(format!(
                "jq filter {filter_src:?} references a denylisted identifier ({denylisted_idents:?}); \
                 custom function definitions and jq's named unbounded-iteration primitives are not allowed"
            ));
        }

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
            .with_global_vars(global_vars.iter().copied())
            .compile(modules)
            .map_err(|err| format!("failed to compile jq filter {filter_src:?}: {err:?}"))?;
        // `arena`/`modules` borrow `filter_src`; `filter` does not, so they
        // drop here while the self-contained program graph lives on.

        Ok(Self {
            src: filter_src.to_owned(),
            filter,
        })
    }

    /// Runs the compiled filter against one JSON input, returning its first
    /// output value. `vars` supplies the values for any global vars the
    /// filter was compiled with (via `NonStdCompat::build`), in the same
    /// order they were declared.
    #[cfg(feature = "nonstd-compat")]
    pub(crate) fn run(
        &self,
        input: serde_json::Value,
        vars: &[serde_json::Value],
    ) -> Result<serde_json::Value, String> {
        use jaq_core::{data, unwrap_valr, Ctx, Vars};
        use jaq_json::Val;

        let input: Val = serde_json::from_value(input).map_err(|err| err.to_string())?;
        let vars: Vec<Val> = vars
            .iter()
            .cloned()
            .map(serde_json::from_value)
            .collect::<Result<_, _>>()
            .map_err(|err| err.to_string())?;
        let ctx = Ctx::<data::JustLut<Val>>::new(&self.filter.lut, Vars::new(vars));

        let output = self
            .filter
            .id
            .run((ctx, input))
            .map(unwrap_valr)
            .next()
            .ok_or_else(|| format!("jq filter {:?} produced no output", self.src))?
            .map_err(|err| format!("jq filter {:?} failed at runtime: {err:?}", self.src))?;

        serde_json::from_str(&output.to_string()).map_err(|err| err.to_string())
    }
}

#[cfg(all(test, feature = "nonstd-compat"))]
mod tests {
    use super::*;

    fn default_denylist() -> Vec<String> {
        DEFAULT_DENYLISTED_IDENTS.iter().map(|s| s.to_string()).collect()
    }

    fn run(filter_src: &str, input: serde_json::Value) -> Result<serde_json::Value, String> {
        CompiledFilter::new(filter_src, &default_denylist(), &["$status"])?.run(input, &[serde_json::json!(200)])
    }

    #[test]
    fn runs_a_simple_filter() {
        let output = run(
            "del(.grant_type)",
            serde_json::json!({
                "grant_type": "client_credentials",
                "client_id": "test",
            }),
        )
        .unwrap();
        assert_eq!(output, serde_json::json!({"client_id": "test"}));
    }

    #[test]
    fn compile_error_is_reported() {
        let err = run("this is not jq", serde_json::json!({})).unwrap_err();
        assert!(err.contains("failed to"), "unexpected error: {err}");
    }

    #[test]
    fn runtime_error_is_reported() {
        let err = run(r#"error("boom")"#, serde_json::json!({})).unwrap_err();
        assert!(
            err.contains("failed at runtime"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn env_is_not_available() {
        let err = run("env", serde_json::json!({})).unwrap_err();
        assert!(
            err.contains("failed to compile"),
            "expected a compile error since `env` should be undefined, got: {err}"
        );
    }

    #[test]
    fn now_is_not_available() {
        let err = run("now", serde_json::json!({})).unwrap_err();
        assert!(
            err.contains("failed to compile"),
            "expected a compile error since `now` should be undefined, got: {err}"
        );
    }

    #[test]
    fn user_defined_functions_are_rejected() {
        let err = run("def f: f; f", serde_json::json!({})).unwrap_err();
        assert!(
            err.contains("denylisted identifier"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn user_defined_functions_nested_in_a_block_are_rejected() {
        // `def` doesn't have to appear at the top level of the filter to be
        // dangerous - it just has to be reachable by the lexer, including
        // inside a parenthesized/bracketed/braced block.
        let err = run("[(def f: f; f)]", serde_json::json!({})).unwrap_err();
        assert!(
            err.contains("denylisted identifier"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn chained_pipe_filters_are_not_falsely_rejected() {
        // Piping through multiple stages (`|`), including update-assignment
        // (`|=`), doesn't reference any denylisted identifier and must not
        // be mistaken for one by `contains_denylisted_ident`.
        let output = run(
            r#".token |= split("\n")[0] | .expiry |= split(".")[0]"#,
            serde_json::json!({"token": "abc\ndef", "expiry": "123.456"}),
        )
        .unwrap();
        assert_eq!(output, serde_json::json!({"token": "abc", "expiry": "123"}));
    }

    #[test]
    fn known_unbounded_loop_patterns_are_rejected_at_compile_time() {
        // Without any custom `def`, these are the standard ways to build an
        // endless computation in jq; `contains_denylisted_ident` catches
        // each by name rather than letting it run and hang the request.
        for filter_src in [
            "reduce repeat(1) as $x (0; .+1)",
            "recurse(.+1)",
            "while(true; .)",
            "until(false; .)",
            "[limit(5; range(0; infinite))]",
        ] {
            let err = run(filter_src, serde_json::json!(0)).unwrap_err();
            assert!(
                err.contains("denylisted identifier"),
                "expected {filter_src:?} to be rejected, got: {err}"
            );
        }
    }

    #[test]
    fn denylist_check_is_heuristic_and_over_rejects() {
        // Documented limitation: a provably-terminating use of a
        // denylisted identifier is still rejected, since the check is
        // purely lexical.
        let err = run("while(. < 10; . + 1)", serde_json::json!(0)).unwrap_err();
        assert!(err.contains("denylisted identifier"));

        // Likewise a non-executable use, like an object-construction key
        // that happens to share a denylisted name.
        let err = run("{repeat: 1}", serde_json::json!({})).unwrap_err();
        assert!(err.contains("denylisted identifier"));
    }

    #[test]
    fn default_denylist_rejects_repeat_without_override() {
        // Exercises the actual public NonStdCompat::default()/new() path (not
        // the `run()`/`default_denylist()` test helpers above), so a
        // regression back to a derived `Default` (which would silently zero
        // `denylisted_idents`, disabling this check for everyone) would show
        // up here as this test *failing to fail*.
        let err = NonStdCompat::new()
            .with_req_map("[limit(3; repeat(1))]")
            .build()
            .unwrap_err();
        assert!(
            err.contains("denylisted identifier"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn with_denylisted_idents_can_widen_to_allow_a_specific_construct() {
        let idents: Vec<String> = DEFAULT_DENYLISTED_IDENTS
            .iter()
            .filter(|&&s| s != "repeat")
            .map(|s| s.to_string())
            .collect();
        let compat = NonStdCompat::new()
            .with_req_map("[limit(3; repeat(1))]")
            .with_denylisted_idents(idents)
            .build()
            .expect("repeat should be allowed once removed from the denylist");
        let output = compat
            .req_map
            .expect("req_map should be compiled")
            .run(serde_json::json!({}), &[])
            .unwrap();
        assert_eq!(output, serde_json::json!([1, 1, 1]));
    }

    #[test]
    fn with_denylisted_idents_empty_disables_the_check_entirely() {
        let compat = NonStdCompat::new()
            .with_req_map("def f: .; f")
            .with_denylisted_idents(Vec::<String>::new())
            .build()
            .expect("empty denylist should allow even `def`");
        let output = compat
            .req_map
            .expect("req_map should be compiled")
            .run(serde_json::json!({"a": 1}), &[])
            .unwrap();
        assert_eq!(output, serde_json::json!({"a": 1}));
    }

    #[test]
    fn compiled_filter_can_be_run_multiple_times() {
        let filter = CompiledFilter::new("{renamed: .value}", &default_denylist(), &[]).unwrap();
        assert_eq!(
            filter.run(serde_json::json!({"value": 1}), &[]).unwrap(),
            serde_json::json!({"renamed": 1})
        );
        assert_eq!(
            filter.run(serde_json::json!({"value": 2}), &[]).unwrap(),
            serde_json::json!({"renamed": 2})
        );
    }
}
