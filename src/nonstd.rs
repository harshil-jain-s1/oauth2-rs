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

/// Compiles and runs a `jq` filter (via the pure-Rust `jaq` crate family)
/// against a single JSON input, returning its first output value.
#[cfg(feature = "nonstd-compat")]
pub(crate) fn run_jq(
    filter_src: &str,
    input: serde_json::Value,
) -> Result<serde_json::Value, String> {
    use jaq_core::load::{Arena, File, Loader};
    use jaq_core::{data, unwrap_valr, Compiler, Ctx, Vars};
    use jaq_json::Val;

    let defs = jaq_core::defs()
        .chain(jaq_std::defs())
        .chain(jaq_json::defs());
    let funs = jaq_core::funs()
        .chain(jaq_std::funs())
        .chain(jaq_json::funs());

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
}
