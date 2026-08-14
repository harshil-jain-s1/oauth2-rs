use crate::{
    AuthType, ClientId, ClientSecret, ErrorResponse, NonStdCompat, RedirectUrl,
    RequestTokenError, Scope, CONTENT_TYPE_FORMENCODED, CONTENT_TYPE_JSON,
};

use base64::prelude::*;
use http::header::{ACCEPT, AUTHORIZATION, CONTENT_TYPE};
use http::{HeaderValue, StatusCode};
use serde::de::DeserializeOwned;
use url::{form_urlencoded, Url};

use std::borrow::Cow;
use std::error::Error;
use std::future::Future;

/// An HTTP request.
pub type HttpRequest = http::Request<Vec<u8>>;

/// An HTTP response.
pub type HttpResponse = http::Response<Vec<u8>>;

/// An asynchronous (future-based) HTTP client.
pub trait AsyncHttpClient<'c> {
    /// Error type returned by HTTP client.
    type Error: Error + 'static;

    /// Future type returned by HTTP client.
    type Future: Future<Output = Result<HttpResponse, Self::Error>> + 'c;

    /// Perform a single HTTP request.
    fn call(&'c self, request: HttpRequest) -> Self::Future;
}
impl<'c, E, F, T> AsyncHttpClient<'c> for T
where
    E: Error + 'static,
    F: Future<Output = Result<HttpResponse, E>> + 'c,
    // We can't implement this for FnOnce because the device authorization flow requires clients to
    // supportmultiple calls.
    T: Fn(HttpRequest) -> F,
{
    type Error = E;
    type Future = F;

    fn call(&'c self, request: HttpRequest) -> Self::Future {
        self(request)
    }
}

/// A synchronous (blocking) HTTP client.
pub trait SyncHttpClient {
    /// Error type returned by HTTP client.
    type Error: Error + 'static;

    /// Perform a single HTTP request.
    fn call(&self, request: HttpRequest) -> Result<HttpResponse, Self::Error>;
}
impl<E, T> SyncHttpClient for T
where
    E: Error + 'static,
    // We can't implement this for FnOnce because the device authorization flow requires clients to
    // support multiple calls.
    T: Fn(HttpRequest) -> Result<HttpResponse, E>,
{
    type Error = E;

    fn call(&self, request: HttpRequest) -> Result<HttpResponse, Self::Error> {
        self(request)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn endpoint_request<'a>(
    auth_type: &'a AuthType,
    client_id: &'a ClientId,
    client_secret: Option<&'a ClientSecret>,
    extra_params: &'a [(Cow<'a, str>, Cow<'a, str>)],
    redirect_url: Option<Cow<'a, RedirectUrl>>,
    scopes: Option<&'a Vec<Cow<'a, Scope>>>,
    url: &'a Url,
    params: Vec<(&'a str, &'a str)>,
    nonstd_compat: Option<&'a NonStdCompat>,
) -> Result<HttpRequest, String> {
    let accept_content_type = nonstd_compat
        .and_then(|c| c.res_type.as_deref())
        .unwrap_or(CONTENT_TYPE_JSON);

    let mut builder = http::Request::builder()
        .uri(url.to_string())
        .method(http::Method::POST)
        .header(
            ACCEPT,
            HeaderValue::from_str(accept_content_type).map_err(|e| e.to_string())?,
        );

    let scopes_opt = scopes.and_then(|scopes| {
        if !scopes.is_empty() {
            Some(
                scopes
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
            )
        } else {
            None
        }
    });

    let mut params: Vec<(&str, &str)> = params;
    if let Some(ref scopes) = scopes_opt {
        params.push(("scope", scopes));
    }

    // FIXME: add support for auth extensions? e.g., client_secret_jwt and private_key_jwt
    match (auth_type, client_secret) {
        // Basic auth only makes sense when a client secret is provided. Otherwise, always pass the
        // client ID in the request body.
        (AuthType::BasicAuth, Some(secret)) => {
            // Section 2.3.1 of RFC 6749 requires separately url-encoding the id and secret
            // before using them as HTTP Basic auth username and password. Note that this is
            // not standard for ordinary Basic auth, so curl won't do it for us.
            let urlencoded_id: String =
                form_urlencoded::byte_serialize(client_id.as_bytes()).collect();
            let urlencoded_secret: String =
                form_urlencoded::byte_serialize(secret.secret().as_bytes()).collect();
            let b64_credential =
                BASE64_STANDARD.encode(format!("{}:{}", &urlencoded_id, urlencoded_secret));
            builder = builder.header(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Basic {}", &b64_credential)).unwrap(),
            );
        }
        (AuthType::RequestBody, _) | (AuthType::BasicAuth, None) => {
            params.push(("client_id", client_id));
            if let Some(client_secret) = client_secret {
                params.push(("client_secret", client_secret.secret()));
            }
        }
    }

    if let Some(ref redirect_url) = redirect_url {
        params.push(("redirect_uri", redirect_url.as_str()));
    }

    params.extend_from_slice(
        extra_params
            .iter()
            .map(|(k, v)| (k.as_ref(), v.as_ref()))
            .collect::<Vec<_>>()
            .as_slice(),
    );

    let req_map = nonstd_compat.and_then(|c| c.req_map.as_ref());

    #[cfg(feature = "nonstd-compat")]
    let (content_type, body): (Cow<'static, str>, Vec<u8>) = match req_map {
        Some(filter) => {
            let json_params = serde_json::Value::Object(
                params
                    .into_iter()
                    .map(|(k, v)| (k.to_string(), serde_json::Value::String(v.to_string())))
                    .collect(),
            );
            let mapped = filter.run(json_params, &[])?;
            req_map_output_to_body(mapped)?
        }
        None => (
            Cow::Borrowed(CONTENT_TYPE_FORMENCODED),
            encode_form_params(params),
        ),
    };
    #[cfg(not(feature = "nonstd-compat"))]
    let (content_type, body): (Cow<'static, str>, Vec<u8>) = {
        let _ = req_map;
        (Cow::Borrowed(CONTENT_TYPE_FORMENCODED), encode_form_params(params))
    };

    builder
        .header(
            CONTENT_TYPE,
            HeaderValue::from_str(&content_type).map_err(|e| e.to_string())?,
        )
        .body(body)
        .map_err(|e| e.to_string())
}

/// Form-urlencodes `pairs` into a request body, the same way for both the
/// standard (no `nonstd_compat`) path and a `req_map` filter that opts into
/// `application/x-www-form-urlencoded` output.
fn encode_form_params<I, K, V>(pairs: I) -> Vec<u8>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    form_urlencoded::Serializer::new(String::new())
        .extend_pairs(pairs)
        .finish()
        .into_bytes()
}

/// Interprets a `req_map` filter's output as `{"content_type": ...,
/// "body": ...}`, returning the header value and the encoded request body.
/// `content_type` defaults to `application/json` if omitted entirely
/// (matching `res_type`'s own default elsewhere in this file), so a filter
/// that only cares about the common JSON case can just return `{body: ...}`:
/// - `application/json`: `body` is serialized directly.
/// - `application/x-www-form-urlencoded`: `body` must be a flat object of
///   string-ish values, encoded via [`encode_form_params`] exactly like the
///   standard (non-`req_map`) path.
/// - anything else: `body` must be a base64-encoded string, decoded and
///   sent verbatim as the request body with `content_type` used as-is for
///   the `Content-Type` header. This lets a filter emit arbitrary
///   bytes/content-types this crate has no built-in knowledge of.
#[cfg(feature = "nonstd-compat")]
fn req_map_output_to_body(mapped: serde_json::Value) -> Result<(Cow<'static, str>, Vec<u8>), String> {
    let serde_json::Value::Object(mut obj) = mapped else {
        return Err(format!("req_map output must be a JSON object, got: {mapped}"));
    };
    let content_type = match obj.remove("content_type") {
        Some(serde_json::Value::String(s)) => Ok(s),
        Some(other) => Err(format!(
            "req_map output field \"content_type\" must be a string, got: {other}"
        )),
        None => Ok(CONTENT_TYPE_JSON.to_string()),
    }?;
    let body_value = obj
        .remove("body")
        .ok_or_else(|| "req_map output missing field \"body\"".to_string())?;

    if content_type == CONTENT_TYPE_JSON {
        let body = serde_json::to_vec(&body_value).map_err(|e| e.to_string())?;
        Ok((Cow::Borrowed(CONTENT_TYPE_JSON), body))
    } else if content_type == CONTENT_TYPE_FORMENCODED {
        let serde_json::Value::Object(body_obj) = &body_value else {
            return Err(format!(
                "req_map \"body\" must be a JSON object when content_type is \
                 {CONTENT_TYPE_FORMENCODED:?}, got: {body_value}"
            ));
        };
        let pairs = body_obj.iter().map(|(k, v)| {
            let v = match v.as_str() {
                Some(s) => Cow::Borrowed(s),
                None => Cow::Owned(v.to_string()),
            };
            (k.as_str(), v)
        });
        Ok((Cow::Borrowed(CONTENT_TYPE_FORMENCODED), encode_form_params(pairs)))
    } else {
        let encoded = body_value.as_str().ok_or_else(|| {
            format!(
                "req_map \"body\" must be a base64-encoded string when content_type is \
                 {content_type:?}, got: {body_value}"
            )
        })?;
        let body = BASE64_STANDARD
            .decode(encoded)
            .map_err(|e| format!("req_map \"body\" is not valid base64: {e}"))?;
        Ok((Cow::Owned(content_type), body))
    }
}

pub(crate) fn endpoint_response<RE, TE, DO>(
    http_response: HttpResponse,
    nonstd_compat: Option<&NonStdCompat>,
) -> Result<DO, RequestTokenError<RE, TE>>
where
    RE: Error,
    TE: ErrorResponse,
    DO: DeserializeOwned,
{
    #[cfg(feature = "nonstd-compat")]
    if let Some(filter) = nonstd_compat.and_then(|c| c.res_map.as_ref()) {
        let expected_content_type = nonstd_compat
            .and_then(|c| c.res_type.as_deref())
            .unwrap_or(CONTENT_TYPE_JSON);
        if http_response.status().is_success() {
            check_response_body(&http_response, expected_content_type)?;
        }

        let response_body = http_response.body().as_slice();
        let value = if is_json_content_type(expected_content_type) {
            serde_json::from_slice(response_body).map_err(|e| {
                RequestTokenError::Other(format!("res_map: response body is not JSON: {e}"))
            })?
        } else {
            serde_json::Value::String(BASE64_STANDARD.encode(response_body))
        };
        let status = serde_json::Value::from(http_response.status().as_u16());
        let mapped = filter.run(value, &[status]).map_err(RequestTokenError::Other)?;

        return if http_response.status().is_success() {
            serde_path_to_error::deserialize(&mapped).map_err(|e| {
                let body = serde_json::to_vec(&mapped).unwrap_or_default();
                RequestTokenError::Parse(e, body)
            })
        } else {
            Err(deserialize_error_response(&mapped))
        };
    }

    check_response_status(&http_response)?;

    let expected_content_type = nonstd_compat
        .and_then(|c| c.res_type.as_deref())
        .unwrap_or(CONTENT_TYPE_JSON);
    check_response_body(&http_response, expected_content_type)?;

    let response_body = http_response.body().as_slice();
    deserialize_response_body(response_body)
}

/// Deserializes `response_body` (raw JSON bytes) into `DO`, the standard
/// (no `res_map` involved) path — shared by the `res_map`-absent branch of
/// [`endpoint_response`] and its `nonstd-compat`-disabled fallback.
fn deserialize_response_body<RE, TE, DO>(
    response_body: &[u8],
) -> Result<DO, RequestTokenError<RE, TE>>
where
    RE: Error,
    TE: ErrorResponse,
    DO: DeserializeOwned,
{
    serde_path_to_error::deserialize(&mut serde_json::Deserializer::from_slice(response_body))
        .map_err(|e| RequestTokenError::Parse(e, response_body.to_vec()))
}

/// Case-insensitively checks whether `content_type` is (a prefix-match for)
/// `application/json`, mirroring [`check_response_body`]'s own comparison
/// style. Used to decide whether `res_map` should receive the raw response
/// bytes parsed as JSON, or a base64-encoded string of those bytes (see
/// [`endpoint_response`]).
#[cfg(feature = "nonstd-compat")]
fn is_json_content_type(content_type: &str) -> bool {
    let content_type = content_type.to_lowercase();
    content_type.starts_with(CONTENT_TYPE_JSON) || content_type.ends_with("+json")
}

pub(crate) fn endpoint_response_status_only<RE, TE>(
    http_response: HttpResponse,
) -> Result<(), RequestTokenError<RE, TE>>
where
    RE: Error + 'static,
    TE: ErrorResponse,
{
    check_response_status(&http_response)
}

fn check_response_status<RE, TE>(
    http_response: &HttpResponse,
) -> Result<(), RequestTokenError<RE, TE>>
where
    RE: Error + 'static,
    TE: ErrorResponse,
{
    if !http_response.status().is_success() {
        let reason = http_response.body().as_slice();
        if reason.is_empty() {
            Err(RequestTokenError::Other(
                "server returned empty error response".to_string(),
            ))
        } else {
            let error = match serde_path_to_error::deserialize::<_, TE>(
                &mut serde_json::Deserializer::from_slice(reason),
            ) {
                Ok(error) => RequestTokenError::ServerResponse(error),
                Err(error) => RequestTokenError::Parse(error, reason.to_vec()),
            };
            Err(error)
        }
    } else {
        Ok(())
    }
}

/// Deserializes an already-mapped (`res_map`-transformed) JSON value into
/// `TE`, mirroring what [`check_response_status`] does for the raw,
/// unmapped error body.
#[cfg(feature = "nonstd-compat")]
fn deserialize_error_response<RE, TE>(value: &serde_json::Value) -> RequestTokenError<RE, TE>
where
    RE: Error,
    TE: ErrorResponse,
{
    match serde_path_to_error::deserialize::<_, TE>(value) {
        Ok(error) => RequestTokenError::ServerResponse(error),
        Err(error) => {
            let body = serde_json::to_vec(value).unwrap_or_default();
            RequestTokenError::Parse(error, body)
        }
    }
}

fn check_response_body<RE, TE>(
    http_response: &HttpResponse,
    expected_content_type: &str,
) -> Result<(), RequestTokenError<RE, TE>>
where
    RE: Error + 'static,
    TE: ErrorResponse,
{
    // Validate that the response Content-Type matches what's expected.
    http_response
    .headers()
    .get(CONTENT_TYPE)
    .map_or(Ok(()), |content_type|
      // Section 3.1.1.1 of RFC 7231 indicates that media types are case-insensitive and
      // may be followed by optional whitespace and/or a parameter (e.g., charset).
      // See https://tools.ietf.org/html/rfc7231#section-3.1.1.1.
      if content_type.to_str().ok().filter(|ct| ct.to_lowercase().starts_with(&expected_content_type.to_lowercase())).is_none() {
        Err(
          RequestTokenError::Other(
            format!(
              "unexpected response Content-Type: {content_type:?}, should be `{expected_content_type}`",
            )
          )
        )
      } else {
        Ok(())
      }
    )?;

    if http_response.body().is_empty() {
        return Err(RequestTokenError::Other(
            "server returned empty response body".to_string(),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::tests::{new_client, FakeError};
    use crate::{AuthorizationCode, TokenResponse};

    use http::{Response, StatusCode};

    #[tokio::test]
    async fn test_async_client_closure() {
        let client = new_client();

        let http_response = Response::builder()
            .status(StatusCode::OK)
            .body(
                "{\"access_token\": \"12/34\", \"token_type\": \"BEARER\"}"
                    .to_string()
                    .into_bytes(),
            )
            .unwrap();

        let token = client
            .exchange_code(AuthorizationCode::new("ccc".to_string()))
            // NB: This tests that the closure doesn't require a static lifetime.
            .request_async(&|_| async { Ok(http_response.clone()) as Result<_, FakeError> })
            .await
            .unwrap();

        assert_eq!("12/34", token.access_token().secret());
    }

    #[cfg(feature = "nonstd-compat")]
    mod nonstd_compat {
        use crate::tests::{new_client, FakeError};
        use crate::{AuthType, NonStdCompat, RequestTokenError, TokenResponse};

        use http::header::CONTENT_TYPE;
        use http::{HeaderValue, Response, StatusCode};

        #[test]
        fn req_map_produces_a_json_body() {
            let client = new_client()
                .set_auth_type(AuthType::RequestBody)
                .set_nonstd_compat(
                    NonStdCompat::new()
                        .with_req_map(
                            "{content_type: \"application/json\", body: del(.grant_type)}",
                        )
                        .build()
                        .unwrap(),
                );

            let http_response = Response::builder()
                .status(StatusCode::OK)
                .header(
                    CONTENT_TYPE,
                    HeaderValue::from_str("application/json").unwrap(),
                )
                .body(
                    "{\"access_token\": \"12/34\", \"token_type\": \"bearer\"}"
                        .to_string()
                        .into_bytes(),
                )
                .unwrap();

            let token = client
                .exchange_client_credentials()
                .request(&move |request: crate::HttpRequest| {
                    assert_eq!(
                        request.headers().get(CONTENT_TYPE).unwrap(),
                        "application/json"
                    );
                    let body: serde_json::Value =
                        serde_json::from_slice(request.body()).unwrap();
                    assert_eq!(
                        body,
                        serde_json::json!({"client_id": "aaa", "client_secret": "bbb"})
                    );
                    Ok(http_response.clone()) as Result<_, FakeError>
                })
                .unwrap();

            assert_eq!("12/34", token.access_token().secret());
        }

        #[test]
        fn req_map_omitting_content_type_defaults_to_json() {
            let client = new_client()
                .set_auth_type(AuthType::RequestBody)
                .set_nonstd_compat(
                    NonStdCompat::new()
                        .with_req_map("{body: del(.grant_type)}")
                        .build()
                        .unwrap(),
                );

            let http_response = Response::builder()
                .status(StatusCode::OK)
                .header(
                    CONTENT_TYPE,
                    HeaderValue::from_str("application/json").unwrap(),
                )
                .body(
                    "{\"access_token\": \"12/34\", \"token_type\": \"bearer\"}"
                        .to_string()
                        .into_bytes(),
                )
                .unwrap();

            let token = client
                .exchange_client_credentials()
                .request(&move |request: crate::HttpRequest| {
                    assert_eq!(
                        request.headers().get(CONTENT_TYPE).unwrap(),
                        "application/json"
                    );
                    let body: serde_json::Value =
                        serde_json::from_slice(request.body()).unwrap();
                    assert_eq!(
                        body,
                        serde_json::json!({"client_id": "aaa", "client_secret": "bbb"})
                    );
                    Ok(http_response.clone()) as Result<_, FakeError>
                })
                .unwrap();

            assert_eq!("12/34", token.access_token().secret());
        }

        #[test]
        fn req_map_can_produce_a_form_encoded_body() {
            let client = new_client().set_auth_type(AuthType::RequestBody).set_nonstd_compat(
                NonStdCompat::new()
                    .with_req_map(
                        "{content_type: \"application/x-www-form-urlencoded\", \
                          body: {clientId: .client_id, secret: .client_secret}}",
                    )
                    .build()
                    .unwrap(),
            );

            let http_response = Response::builder()
                .status(StatusCode::OK)
                .header(
                    CONTENT_TYPE,
                    HeaderValue::from_str("application/json").unwrap(),
                )
                .body(
                    "{\"access_token\": \"12/34\", \"token_type\": \"bearer\"}"
                        .to_string()
                        .into_bytes(),
                )
                .unwrap();

            let token = client
                .exchange_client_credentials()
                .request(&move |request: crate::HttpRequest| {
                    assert_eq!(
                        request.headers().get(CONTENT_TYPE).unwrap(),
                        "application/x-www-form-urlencoded"
                    );
                    assert_eq!(
                        String::from_utf8(request.body().to_owned()).unwrap(),
                        "clientId=aaa&secret=bbb"
                    );
                    Ok(http_response.clone()) as Result<_, FakeError>
                })
                .unwrap();

            assert_eq!("12/34", token.access_token().secret());
        }

        #[test]
        fn req_map_arbitrary_content_type_base64_decodes_body() {
            let client = new_client().set_auth_type(AuthType::RequestBody).set_nonstd_compat(
                NonStdCompat::new()
                    .with_req_map(
                        "{content_type: \"application/octet-stream\", body: (.client_id | @base64)}",
                    )
                    .build()
                    .unwrap(),
            );

            let http_response = Response::builder()
                .status(StatusCode::OK)
                .header(
                    CONTENT_TYPE,
                    HeaderValue::from_str("application/json").unwrap(),
                )
                .body(
                    "{\"access_token\": \"12/34\", \"token_type\": \"bearer\"}"
                        .to_string()
                        .into_bytes(),
                )
                .unwrap();

            let token = client
                .exchange_client_credentials()
                .request(&move |request: crate::HttpRequest| {
                    assert_eq!(
                        request.headers().get(CONTENT_TYPE).unwrap(),
                        "application/octet-stream"
                    );
                    assert_eq!(request.body().as_slice(), b"aaa");
                    Ok(http_response.clone()) as Result<_, FakeError>
                })
                .unwrap();

            assert_eq!("12/34", token.access_token().secret());
        }

        #[test]
        fn req_map_body_must_be_a_string_for_unknown_content_type() {
            let client = new_client().set_nonstd_compat(
                NonStdCompat::new()
                    .with_req_map("{content_type: \"application/octet-stream\", body: 42}")
                    .build()
                    .unwrap(),
            );

            let err = client
                .exchange_client_credentials()
                .request(&move |_: crate::HttpRequest| -> Result<_, FakeError> {
                    unreachable!("request should not be sent when body isn't base64-able")
                })
                .unwrap_err();

            match err {
                RequestTokenError::Other(msg) => {
                    assert!(msg.contains("base64-encoded string"), "unexpected error: {msg}")
                }
                other => panic!("unexpected error variant: {other:?}"),
            }
        }

        #[test]
        fn req_map_body_invalid_base64_errors() {
            let client = new_client().set_nonstd_compat(
                NonStdCompat::new()
                    .with_req_map(
                        "{content_type: \"application/octet-stream\", body: \"not valid base64!\"}",
                    )
                    .build()
                    .unwrap(),
            );

            let err = client
                .exchange_client_credentials()
                .request(&move |_: crate::HttpRequest| -> Result<_, FakeError> {
                    unreachable!("request should not be sent when body isn't valid base64")
                })
                .unwrap_err();

            match err {
                RequestTokenError::Other(msg) => {
                    assert!(msg.contains("not valid base64"), "unexpected error: {msg}")
                }
                other => panic!("unexpected error variant: {other:?}"),
            }
        }

        #[test]
        fn res_map_base64_decodes_non_json_response_body() {
            let client = new_client().set_nonstd_compat(
                NonStdCompat::new()
                    .with_res_type("application/octet-stream")
                    .with_res_map("{access_token: (. | @base64d), token_type: \"bearer\"}")
                    .build()
                    .unwrap(),
            );

            let raw_body = b"opaque-token-value".to_vec();
            let http_response = Response::builder()
                .status(StatusCode::OK)
                .header(
                    CONTENT_TYPE,
                    HeaderValue::from_str("application/octet-stream").unwrap(),
                )
                .body(raw_body.clone())
                .unwrap();

            let token = client
                .exchange_client_credentials()
                .request(&move |_| Ok(http_response.clone()) as Result<_, FakeError>)
                .unwrap();

            assert_eq!(token.access_token().secret(), "opaque-token-value");
        }

        #[test]
        fn res_map_transforms_response_before_deserializing() {
            let client = new_client().set_nonstd_compat(
                NonStdCompat::new()
                    .with_res_map("{access_token: .jwt, token_type: \"bearer\"}")
                    .build()
                    .unwrap(),
            );

            let http_response = Response::builder()
                .status(StatusCode::OK)
                .header(
                    CONTENT_TYPE,
                    HeaderValue::from_str("application/json").unwrap(),
                )
                .body("{\"jwt\": \"12/34\"}".to_string().into_bytes())
                .unwrap();

            let token = client
                .exchange_client_credentials()
                .request(&move |_| Ok(http_response.clone()) as Result<_, FakeError>)
                .unwrap();

            assert_eq!("12/34", token.access_token().secret());
        }

        #[test]
        fn res_map_reshapes_nonstandard_error_body_using_status() {
            use crate::basic::BasicErrorResponseType;
            use crate::ErrorResponse;

            let client = new_client().set_nonstd_compat(
                NonStdCompat::new()
                    .with_res_map(
                        "if $status == 200 then {access_token: .jwt, token_type: \"bearer\"} \
                         else {error: \"invalid_client\", error_description: .msg} end",
                    )
                    .build()
                    .unwrap(),
            );

            // Non-standard error body: no top-level `error` field, so today
            // (without `res_map` mapping error responses too) this would
            // fail to deserialize into `BasicErrorResponse` and surface as
            // `RequestTokenError::Parse`.
            let error_response = Response::builder()
                .status(StatusCode::UNAUTHORIZED)
                .header(
                    CONTENT_TYPE,
                    HeaderValue::from_str("application/json").unwrap(),
                )
                .body(
                    "{\"msg\": \"bad creds\"}"
                        .to_string()
                        .into_bytes(),
                )
                .unwrap();

            let err = client
                .exchange_client_credentials()
                .request(&move |_| Ok(error_response.clone()) as Result<_, FakeError>)
                .unwrap_err();

            match err {
                RequestTokenError::ServerResponse(e) => {
                    assert_eq!(e.error(), &BasicErrorResponseType::InvalidClient);
                    assert_eq!(e.error_description(), Some(&"bad creds".to_string()));
                }
                other => panic!("expected ServerResponse, got {other:?}"),
            }

            // The success arm of the same filter/client still works,
            // confirming one filter branches correctly both ways via
            // `$status`.
            let success_response = Response::builder()
                .status(StatusCode::OK)
                .header(
                    CONTENT_TYPE,
                    HeaderValue::from_str("application/json").unwrap(),
                )
                .body("{\"jwt\": \"12/34\"}".to_string().into_bytes())
                .unwrap();

            let token = client
                .exchange_client_credentials()
                .request(&move |_| Ok(success_response.clone()) as Result<_, FakeError>)
                .unwrap();

            assert_eq!("12/34", token.access_token().secret());
        }

        #[test]
        fn res_type_allows_a_non_json_content_type() {
            let client = new_client().set_nonstd_compat(
                NonStdCompat::new().with_res_type("application/vnd.provider+json"),
            );

            let http_response = Response::builder()
                .status(StatusCode::OK)
                .header(
                    CONTENT_TYPE,
                    HeaderValue::from_str("application/vnd.provider+json").unwrap(),
                )
                .body(
                    "{\"access_token\": \"12/34\", \"token_type\": \"bearer\"}"
                        .to_string()
                        .into_bytes(),
                )
                .unwrap();

            let token = client
                .exchange_client_credentials()
                .request(&move |request: crate::HttpRequest| {
                    assert_eq!(
                        request.headers().get(http::header::ACCEPT).unwrap(),
                        "application/vnd.provider+json"
                    );
                    Ok(http_response.clone()) as Result<_, FakeError>
                })
                .unwrap();

            assert_eq!("12/34", token.access_token().secret());
        }

        #[test]
        fn res_map_treats_a_vendor_json_content_type_as_json() {
            // A `res_type` of `application/vnd.provider+json` is JSON (per
            // RFC 6839's `+json` structured-syntax suffix), so a `res_map`
            // filter should receive it parsed as JSON, not base64-encoded.
            let client = new_client().set_nonstd_compat(
                NonStdCompat::new()
                    .with_res_type("application/vnd.provider+json")
                    .with_res_map("{access_token: .jwt, token_type: \"bearer\"}")
                    .build()
                    .unwrap(),
            );

            let http_response = Response::builder()
                .status(StatusCode::OK)
                .header(
                    CONTENT_TYPE,
                    HeaderValue::from_str("application/vnd.provider+json").unwrap(),
                )
                .body("{\"jwt\": \"12/34\"}".to_string().into_bytes())
                .unwrap();

            let token = client
                .exchange_client_credentials()
                .request(&move |_| Ok(http_response.clone()) as Result<_, FakeError>)
                .unwrap();

            assert_eq!("12/34", token.access_token().secret());
        }

        #[test]
        fn invalid_filter_fails_at_build_time() {
            // Compile errors now surface as soon as `build()` runs, before the
            // filter is ever attached to a `Client` - no network/request
            // machinery involved at all.
            let err = NonStdCompat::new()
                .with_req_map("this is not jq")
                .build()
                .unwrap_err();

            assert!(err.contains("failed to"), "unexpected error: {err}");
        }

        #[test]
        fn req_map_runtime_error_surfaces_as_other() {
            // A filter that compiles fine but fails at execution time (e.g.
            // explicitly, via `error(...)`) should still surface through the
            // normal request-preparation error path.
            let client = new_client().set_nonstd_compat(
                NonStdCompat::new()
                    .with_req_map(r#"error("boom")"#)
                    .build()
                    .unwrap(),
            );

            let err = client
                .exchange_client_credentials()
                .request(&move |_: crate::HttpRequest| -> Result<_, FakeError> {
                    unreachable!("request should not be sent when req_map fails at runtime")
                })
                .unwrap_err();

            assert!(matches!(err, RequestTokenError::Other(_)));
        }
    }
}
