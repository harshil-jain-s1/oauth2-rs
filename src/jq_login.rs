//! A fully config-driven JSON-body login flow, used by providers that
//! authenticate via a single JSON POST request rather than a standard OAuth2
//! grant, e.g.:
//!
//! ```sh
//! curl --location 'https://example.com/login' \
//!   --header 'Content-Type: application/json' \
//!   --data '{"client_id":"test","client_secret":"secret"}'
//! ```
//!
//! responding with something like:
//!
//! ```json
//! {
//!   "access_token": "eyJhbG....",
//!   "refresh_token": "0e42b0b8....",
//!   "expires_in": 86400
//! }
//! ```
//!
//! There's no `grant_type`, the request body is JSON rather than
//! form-urlencoded, and the field names, extra static parameters, and even
//! which fields exist at all are entirely provider-specific. None of the
//! crate's built-in grants (`exchange_code`/`exchange_password`/
//! `exchange_client_credentials`/`exchange_refresh_token`) can produce this
//! shape of request, so [`JqLoginClient`] builds the request/response by hand
//! using the crate's public [`HttpRequest`]/[`HttpResponse`]/
//! [`SyncHttpClient`]/[`AsyncHttpClient`] types.
//!
//! Everything about the shape of the login exchange is driven by
//! [`JqLoginConfig`]:
//! - which JSON field (if any) carries the client id / secret in the request
//! - a `jq` filter (`request_filter`) that turns `{"client_id": ...,
//!   "client_secret": ...}` into the JSON body to send, e.g. renaming
//!   fields, dropping one of them, or merging in extra static parameters
//!   like a `grant_type` some providers still expect
//! - a `jq` filter (`response_filter`) that turns the raw JSON response into
//!   `{"access_token": ..., "refresh_token": ..., "expires_in": ...,
//!   "token_type": ..., "scope": ...}`, so arbitrary provider-specific field
//!   names, nesting, and type coercions (e.g. a numeric expiry sent as a
//!   string) are handled in `jq` rather than as one-off Rust config knobs
//!
//! `request_filter`'s input is always exactly `{"client_id": ...,
//! "client_secret": ...}` - nothing else is supplied automatically - so any
//! other field a provider's login endpoint expects has to be added by the
//! filter itself, as a literal merged into its output object. In practice
//! that covers the full range of things such an endpoint might ask for:
//! - `client_id` / `client_secret`: pulled straight from the filter input,
//!   possibly under different field names, e.g. `{clientId: .client_id}`
//! - `grant_type`: a static literal, e.g.
//!   `{grant_type: "client_credentials"}`
//! - `scope`: a static, space-delimited string or JSON array literal, e.g.
//!   `{scope: "read write"}` or `{scope: ["read", "write"]}`
//! - any other provider-specific static parameter (`audience`, `resource`,
//!   an API version/tenant id, ...): also just a literal, e.g.
//!   `{audience: "https://api.example.com"}`
//!
//! Only `client_id`/`client_secret` placement in headers or HTTP Basic auth
//! (as opposed to the body) is controlled by [`CredentialPlacement`] instead
//! of the filter, since that's outside the JSON body `request_filter`
//! produces.
//!
//! Every response field except the access token is optional: if the
//! response filter's output doesn't include a field, parsing simply leaves
//! that piece unset instead of failing. Only a missing access token is
//! treated as an error, since there is nothing usable to return without one.
//!
//! Filters run in-process via the [`jaq`](https://github.com/01mf02/jaq)
//! crate family, a pure-Rust jq implementation, rather than shelling out to
//! a system `jq` binary.
//!
//! Requires the `jq-login` feature.

use std::time::Duration;

use base64::prelude::*;
use jaq_core::load::{Arena, File, Loader};
use jaq_core::{data, unwrap_valr, Compiler, Ctx, Vars};
use jaq_json::Val;
use serde::Deserialize;
use url::Url;

use crate::basic::BasicTokenType;
use crate::{
    AccessToken, AsyncHttpClient, ClientId, ClientSecret, EmptyExtraTokenFields, HttpRequest,
    HttpResponse, RefreshToken, Scope, StandardTokenResponse, SyncHttpClient,
};
#[cfg(test)]
use crate::TokenResponse;

/// Where/how the client id and secret are sent on the login request. Covers
/// the three common ways a JQ login endpoint expects credentials.
/// Deserializes from a plain string tag, e.g. `"body"`, `{"headers": {...}}`
/// or `"basic_auth"`.
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CredentialPlacement {
    /// Expose `{"client_id": ..., "client_secret": ...}` as the input to
    /// `request_filter`, which decides how (or whether) they end up in the
    /// JSON request body.
    Body,
    /// Send the id/secret as separate request headers with the given names.
    /// Either header can be `None` to omit it. `request_filter` is still run
    /// with `{"client_id": ..., "client_secret": ...}` as input, but is
    /// expected not to place the credentials in the body in this mode.
    Headers {
        /// Header name carrying the client id, if any.
        client_id_header: Option<String>,
        /// Header name carrying the client secret, if any.
        secret_header: Option<String>,
    },
    /// Send the id/secret via HTTP Basic authentication, i.e. an
    /// `Authorization: Basic base64(client_id:secret)` header, per
    /// [RFC 7617](https://tools.ietf.org/html/rfc7617). `request_filter` is
    /// still run with `{"client_id": ..., "client_secret": ...}` as input,
    /// but is expected not to place the credentials in the body in this
    /// mode.
    BasicAuth,
}

/// Fully describes the shape of a provider's JSON login request/response, so
/// [`JqLoginClient`] isn't hardcoded to any single provider's fields. See
/// the module docs above for the JSON config format this deserializes from.
#[derive(Debug, Clone, Deserialize)]
pub struct JqLoginConfig {
    /// How the client id/secret are attached to the request.
    pub credentials: CredentialPlacement,
    /// A `jq` filter run against `{"client_id": ..., "client_secret": ...}`,
    /// producing the JSON object sent as the request body. Static extra
    /// parameters (e.g. a `grant_type` some providers still expect) are
    /// just literals in the filter, e.g. `{grant_type: "client_credentials"}
    /// + .`.
    pub request_filter: String,
    /// A `jq` filter run against the raw JSON response, producing
    /// `{"access_token": ..., "refresh_token": ..., "expires_in": ...,
    /// "token_type": ..., "scope": ...}`. Only `access_token` is required;
    /// the rest may be omitted from the filter's output.
    pub response_filter: String,
}

/// A login client for providers that authenticate via a single JSON POST
/// request carrying a client id/secret and returning a bearer token, rather
/// than a standard OAuth2 grant.
#[derive(Debug, Clone)]
pub struct JqLoginClient {
    login_url: Url,
    client_id: ClientId,
    client_secret: ClientSecret,
    config: JqLoginConfig,
}

impl JqLoginClient {
    /// Builds a client for the login endpoint at `login_url`, authenticating
    /// with `client_id`/`client_secret`, per `config`'s request/response
    /// shape.
    pub fn new(
        login_url: Url,
        client_id: ClientId,
        client_secret: ClientSecret,
        config: JqLoginConfig,
    ) -> Self {
        Self {
            login_url,
            client_id,
            client_secret,
            config,
        }
    }

    /// Performs the login request synchronously and returns the parsed
    /// token response.
    pub fn login(
        &self,
        http_client: &impl SyncHttpClient,
    ) -> Result<StandardTokenResponse<EmptyExtraTokenFields, BasicTokenType>, anyhow::Error> {
        let request = self.prepare_login_request()?;
        let response = http_client
            .call(request)
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        self.parse_response(&response)
    }

    /// Performs the login request asynchronously and returns the parsed
    /// token response.
    pub async fn login_async<'c>(
        &self,
        http_client: &'c impl AsyncHttpClient<'c>,
    ) -> Result<StandardTokenResponse<EmptyExtraTokenFields, BasicTokenType>, anyhow::Error> {
        let request = self.prepare_login_request()?;
        let response = http_client
            .call(request)
            .await
            .map_err(|err| anyhow::anyhow!("{err}"))?;
        self.parse_response(&response)
    }

    fn prepare_login_request(&self) -> Result<HttpRequest, anyhow::Error> {
        let credentials_input = serde_json::json!({
            "client_id": self.client_id.as_str(),
            "client_secret": self.client_secret.secret(),
        });
        let body = run_jq(&self.config.request_filter, credentials_input)?;
        self.build_request(body)
    }

    fn build_request(&self, body: serde_json::Value) -> Result<HttpRequest, anyhow::Error> {
        let mut builder = http::Request::builder()
            .method(http::Method::POST)
            .uri(self.login_url.as_str())
            .header(http::header::CONTENT_TYPE, "application/json")
            .header(http::header::ACCEPT, "application/json");

        match &self.config.credentials {
            CredentialPlacement::Body => {}
            CredentialPlacement::Headers {
                client_id_header,
                secret_header,
            } => {
                if let Some(header) = client_id_header {
                    builder = builder.header(header, self.client_id.as_str());
                }
                if let Some(header) = secret_header {
                    builder = builder.header(header, self.client_secret.secret());
                }
            }
            CredentialPlacement::BasicAuth => {
                let credentials = format!(
                    "{}:{}",
                    self.client_id.as_str(),
                    self.client_secret.secret()
                );
                builder = builder.header(
                    http::header::AUTHORIZATION,
                    format!("Basic {}", BASE64_STANDARD.encode(credentials)),
                );
            }
        }

        Ok(builder.body(serde_json::to_vec(&body)?)?)
    }

    fn parse_response(
        &self,
        response: &HttpResponse,
    ) -> Result<StandardTokenResponse<EmptyExtraTokenFields, BasicTokenType>, anyhow::Error> {
        let raw_body: serde_json::Value = serde_json::from_slice(response.body())?;
        let body = run_jq(&self.config.response_filter, raw_body)?;

        let access_token = body
            .get("access_token")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                anyhow::anyhow!("response_filter output missing string field \"access_token\"")
            })?;

        let token_type = body
            .get("token_type")
            .and_then(serde_json::Value::as_str)
            .map(parse_token_type)
            .unwrap_or(BasicTokenType::Bearer);

        let mut token_response = StandardTokenResponse::new(
            AccessToken::new(access_token.to_string()),
            token_type,
            EmptyExtraTokenFields {},
        );

        if let Some(refresh_token) = body
            .get("refresh_token")
            .and_then(serde_json::Value::as_str)
        {
            token_response.set_refresh_token(Some(RefreshToken::new(refresh_token.to_string())));
        }

        if let Some(expires_in) = body.get("expires_in").and_then(serde_json::Value::as_u64) {
            token_response.set_expires_in(Some(&Duration::from_secs(expires_in)));
        }

        if let Some(scopes) = body.get("scope").and_then(parse_scopes) {
            token_response.set_scopes(Some(scopes));
        }

        Ok(token_response)
    }
}

/// Compiles and runs a `jq` filter (via the pure-Rust `jaq` crate family)
/// against a single JSON input, returning its first output value. Filters
/// are expected to always produce exactly one object.
fn run_jq(filter_src: &str, input: serde_json::Value) -> Result<serde_json::Value, anyhow::Error> {
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
        .map_err(|err| anyhow::anyhow!("failed to parse jq filter {filter_src:?}: {err:?}"))?;

    let filter = Compiler::default()
        .with_funs(funs)
        .compile(modules)
        .map_err(|err| anyhow::anyhow!("failed to compile jq filter {filter_src:?}: {err:?}"))?;

    let input: Val = serde_json::from_value(input)?;
    let ctx = Ctx::<data::JustLut<Val>>::new(&filter.lut, Vars::new([]));

    let output = filter
        .id
        .run((ctx, input))
        .map(unwrap_valr)
        .next()
        .ok_or_else(|| anyhow::anyhow!("jq filter {filter_src:?} produced no output"))?
        .map_err(|err| anyhow::anyhow!("jq filter {filter_src:?} failed at runtime: {err:?}"))?;

    Ok(serde_json::from_str(&output.to_string())?)
}

/// Case-insensitively maps a token type string onto `BasicTokenType`,
/// falling back to `Extension` for anything unrecognized rather than
/// failing.
fn parse_token_type(value: &str) -> BasicTokenType {
    match value.to_ascii_lowercase().as_str() {
        "bearer" => BasicTokenType::Bearer,
        "mac" => BasicTokenType::Mac,
        _ => BasicTokenType::Extension(value.to_string()),
    }
}

/// Reads scopes from a JSON array of strings, or a single string split on
/// whitespace.
fn parse_scopes(value: &serde_json::Value) -> Option<Vec<Scope>> {
    if let Some(values) = value.as_array() {
        return Some(
            values
                .iter()
                .filter_map(serde_json::Value::as_str)
                .map(|s| Scope::new(s.to_string()))
                .collect(),
        );
    }
    value.as_str().map(|s| {
        s.split_whitespace()
            .map(|s| Scope::new(s.to_string()))
            .collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_config() -> JqLoginConfig {
        JqLoginConfig {
            credentials: CredentialPlacement::Body,
            request_filter: "{client_id: .client_id, client_secret: .client_secret}".to_string(),
            response_filter: "{access_token: .access_token, refresh_token: .refresh_token, \
                               expires_in: .expires_in}"
                .to_string(),
        }
    }

    fn client_with(config: JqLoginConfig) -> JqLoginClient {
        JqLoginClient::new(
            Url::parse("https://example.com/login").unwrap(),
            ClientId::new("test".to_string()),
            ClientSecret::new("secret".to_string()),
            config,
        )
    }

    fn json_response(value: serde_json::Value) -> HttpResponse {
        http::Response::builder()
            .status(200)
            .body(serde_json::to_vec(&value).unwrap())
            .unwrap()
    }

    #[test]
    fn parses_full_json_login_response() {
        let client = client_with(base_config());

        let response = json_response(serde_json::json!({
            "access_token": "eyJhbG....",
            "refresh_token": "0e42b0b8....",
            "expires_in": 86400
        }));

        let token_response = client.parse_response(&response).unwrap();

        assert_eq!(token_response.access_token().secret(), "eyJhbG....");
        assert_eq!(
            token_response.refresh_token().unwrap().secret(),
            "0e42b0b8...."
        );
        assert_eq!(
            token_response.expires_in(),
            Some(Duration::from_secs(86400))
        );
    }

    #[test]
    fn tolerates_missing_optional_fields() {
        // A response that only contains an access token still parses fine;
        // fields absent from the response_filter's output are simply left
        // unset.
        let client = client_with(base_config());

        let response = json_response(serde_json::json!({
            "access_token": "eyJhbG...."
        }));

        let token_response = client.parse_response(&response).unwrap();

        assert_eq!(token_response.access_token().secret(), "eyJhbG....");
        assert!(token_response.refresh_token().is_none());
        assert_eq!(token_response.expires_in(), None);
        assert_eq!(*token_response.token_type(), BasicTokenType::Bearer);
    }

    #[test]
    fn errors_only_when_access_token_itself_is_missing() {
        let client = client_with(base_config());

        let response = json_response(serde_json::json!({
            "refresh_token": "0e42b0b8...."
        }));

        assert!(client.parse_response(&response).is_err());
    }

    #[test]
    fn supports_arbitrary_field_names_and_extra_request_fields() {
        // Demonstrates the same client working against a provider using
        // completely different field names, a numeric-string expiry, extra
        // static request parameters, and array-style scopes, all expressed
        // via jq filters rather than Rust config fields.
        let config = JqLoginConfig {
            credentials: CredentialPlacement::Body,
            request_filter: "{clientId: .client_id, secret: .client_secret, \
                              grant_type: \"client_login\"}"
                .to_string(),
            response_filter: "{access_token: .jwt, refresh_token: .refreshToken, \
                               expires_in: (.expiresIn | tonumber), token_type: .tokenType, \
                               scope: .scopes}"
                .to_string(),
        };

        let client = client_with(config);

        let request = client.prepare_login_request().unwrap();
        let request_body: serde_json::Value = serde_json::from_slice(request.body()).unwrap();
        assert_eq!(request_body["clientId"], "test");
        assert_eq!(request_body["secret"], "secret");
        assert_eq!(request_body["grant_type"], "client_login");

        let response = json_response(serde_json::json!({
            "jwt": "eyJhbG....",
            "refreshToken": "0e42b0b8....",
            "expiresIn": "86400",
            "tokenType": "BEARER",
            "scopes": ["read", "write"]
        }));

        let token_response = client.parse_response(&response).unwrap();

        assert_eq!(token_response.access_token().secret(), "eyJhbG....");
        assert_eq!(
            token_response.expires_in(),
            Some(Duration::from_secs(86400))
        );
        assert_eq!(*token_response.token_type(), BasicTokenType::Bearer);
        assert_eq!(
            token_response.scopes().unwrap(),
            &vec![
                Scope::new("read".to_string()),
                Scope::new("write".to_string())
            ]
        );
    }

    #[test]
    fn supports_static_grant_type_and_extra_oauth_fields_in_request() {
        // request_filter can merge in arbitrary static OAuth-style
        // parameters a provider's JSON login endpoint expects alongside the
        // credentials, e.g. grant_type/audience/scope, purely as jq
        // literals.
        let mut config = base_config();
        config.request_filter = "{client_id: .client_id, client_secret: .client_secret, \
                                   grant_type: \"client_credentials\", \
                                   audience: \"https://api.example.com\", \
                                   scope: \"read write\"}"
            .to_string();
        let client = client_with(config);

        let request = client.prepare_login_request().unwrap();
        let request_body: serde_json::Value = serde_json::from_slice(request.body()).unwrap();
        assert_eq!(request_body["client_id"], "test");
        assert_eq!(request_body["client_secret"], "secret");
        assert_eq!(request_body["grant_type"], "client_credentials");
        assert_eq!(request_body["audience"], "https://api.example.com");
        assert_eq!(request_body["scope"], "read write");
    }

    #[test]
    fn parses_access_and_refresh_token_from_delimited_string() {
        // Some providers return both tokens concatenated in a single field
        // instead of separate ones; response_filter can split on the
        // delimiter to pull each one out.
        let mut config = base_config();
        config.response_filter = "{access_token: (.tokens | split(\":\") | .[0]), \
                                    refresh_token: (.tokens | split(\":\") | .[1])}"
            .to_string();
        let client = client_with(config);

        let response = json_response(serde_json::json!({
            "tokens": "eyJhbG....:0e42b0b8...."
        }));

        let token_response = client.parse_response(&response).unwrap();

        assert_eq!(token_response.access_token().secret(), "eyJhbG....");
        assert_eq!(
            token_response.refresh_token().unwrap().secret(),
            "0e42b0b8...."
        );
    }

    #[test]
    fn parses_access_and_refresh_token_from_two_element_array() {
        // Some providers return both tokens as a 2-element JSON array
        // instead of separate fields; response_filter indexes into it.
        let mut config = base_config();
        config.response_filter =
            "{access_token: .tokens[0], refresh_token: .tokens[1]}".to_string();
        let client = client_with(config);

        let response = json_response(serde_json::json!({
            "tokens": ["eyJhbG....", "0e42b0b8...."]
        }));

        let token_response = client.parse_response(&response).unwrap();

        assert_eq!(token_response.access_token().secret(), "eyJhbG....");
        assert_eq!(
            token_response.refresh_token().unwrap().secret(),
            "0e42b0b8...."
        );
    }

    #[test]
    fn sends_credentials_as_custom_headers() {
        let mut config = base_config();
        config.credentials = CredentialPlacement::Headers {
            client_id_header: Some("X-Client-Id".to_string()),
            secret_header: Some("X-Client-Secret".to_string()),
        };
        config.request_filter = "{}".to_string();
        let client = client_with(config);

        let request = client.prepare_login_request().unwrap();

        assert_eq!(request.headers().get("X-Client-Id").unwrap(), "test");
        assert_eq!(request.headers().get("X-Client-Secret").unwrap(), "secret");
        // Credentials should not leak into the body in this mode.
        let request_body: serde_json::Value = serde_json::from_slice(request.body()).unwrap();
        assert_eq!(request_body, serde_json::json!({}));
    }

    #[test]
    fn sends_credentials_via_http_basic_auth() {
        let mut config = base_config();
        config.credentials = CredentialPlacement::BasicAuth;
        config.request_filter = "{}".to_string();
        let client = client_with(config);

        let request = client.prepare_login_request().unwrap();

        let expected = format!("Basic {}", BASE64_STANDARD.encode("test:secret"));
        assert_eq!(
            request.headers().get(http::header::AUTHORIZATION).unwrap(),
            &expected
        );
        // Credentials should not leak into the body in this mode.
        let request_body: serde_json::Value = serde_json::from_slice(request.body()).unwrap();
        assert_eq!(request_body, serde_json::json!({}));
    }

    #[test]
    fn deserializes_config_from_json() {
        let config: JqLoginConfig = serde_json::from_value(serde_json::json!({
            "credentials": "body",
            "request_filter": "{client_id: .client_id, client_secret: .client_secret}",
            "response_filter": "{access_token: .access_token, refresh_token: .refresh_token, expires_in: .expires_in}"
        }))
        .unwrap();

        assert!(matches!(config.credentials, CredentialPlacement::Body));
        assert!(config.response_filter.contains("access_token"));
    }

    #[test]
    fn deserializes_basic_auth_config_from_json() {
        let config: JqLoginConfig = serde_json::from_value(serde_json::json!({
            "credentials": "basic_auth",
            "request_filter": "{}",
            "response_filter": "{access_token: .jwt}"
        }))
        .unwrap();

        assert!(matches!(config.credentials, CredentialPlacement::BasicAuth));
    }
}
