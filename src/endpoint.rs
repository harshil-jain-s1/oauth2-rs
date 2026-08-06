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

    let req_map = nonstd_compat.and_then(|c| c.req_map.as_deref());

    #[cfg(feature = "nonstd-compat")]
    if let Some(filter) = req_map {
        let json_params = serde_json::Value::Object(
            params
                .into_iter()
                .map(|(k, v)| (k.to_string(), serde_json::Value::String(v.to_string())))
                .collect(),
        );
        let mapped = crate::nonstd::run_jq(filter, json_params)?;
        let body = serde_json::to_vec(&mapped).map_err(|e| e.to_string())?;
        return builder
            .header(CONTENT_TYPE, HeaderValue::from_static(CONTENT_TYPE_JSON))
            .body(body)
            .map_err(|e| e.to_string());
    }
    #[cfg(not(feature = "nonstd-compat"))]
    let _ = req_map;

    let body = form_urlencoded::Serializer::new(String::new())
        .extend_pairs(params)
        .finish()
        .into_bytes();

    builder
        .header(
            CONTENT_TYPE,
            HeaderValue::from_static(CONTENT_TYPE_FORMENCODED),
        )
        .body(body)
        .map_err(|e| e.to_string())
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
    check_response_status(&http_response)?;

    let expected_content_type = nonstd_compat
        .and_then(|c| c.res_type.as_deref())
        .unwrap_or(CONTENT_TYPE_JSON);
    check_response_body(&http_response, expected_content_type)?;

    let response_body = http_response.body().as_slice();

    #[cfg(feature = "nonstd-compat")]
    let mapped_body: Option<Vec<u8>> = match nonstd_compat.and_then(|c| c.res_map.as_deref()) {
        Some(filter) => {
            let value: serde_json::Value = serde_json::from_slice(response_body).map_err(|e| {
                RequestTokenError::Other(format!("res_map: response body is not JSON: {e}"))
            })?;
            let mapped = crate::nonstd::run_jq(filter, value).map_err(RequestTokenError::Other)?;
            Some(serde_json::to_vec(&mapped).map_err(|e| {
                RequestTokenError::Other(format!("res_map: failed to re-serialize: {e}"))
            })?)
        }
        None => None,
    };
    #[cfg(not(feature = "nonstd-compat"))]
    let mapped_body: Option<Vec<u8>> = None;

    let response_body = mapped_body.as_deref().unwrap_or(response_body);

    serde_path_to_error::deserialize(&mut serde_json::Deserializer::from_slice(response_body))
        .map_err(|e| RequestTokenError::Parse(e, response_body.to_vec()))
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
                .set_nonstd_compat(NonStdCompat::new().with_req_map("del(.grant_type)"));

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
        fn res_map_transforms_response_before_deserializing() {
            let client = new_client().set_nonstd_compat(
                NonStdCompat::new()
                    .with_res_map("{access_token: .jwt, token_type: \"bearer\"}"),
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
        fn jq_error_surfaces_as_other() {
            let client = new_client().set_nonstd_compat(
                NonStdCompat::new().with_req_map("this is not jq"),
            );

            let err = client
                .exchange_client_credentials()
                .request(&move |_: crate::HttpRequest| -> Result<_, FakeError> {
                    unreachable!("request should not be sent when req_map fails to compile")
                })
                .unwrap_err();

            assert!(matches!(err, RequestTokenError::Other(_)));
        }
    }
}
