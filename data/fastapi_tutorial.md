# FastAPI Route and Auth Tutorial

FastAPI is a fast, asynchronous Python web framework built on top of Starlette and Pydantic. It provides automatic OpenAPI documentation out-of-the-box.

## Dependency Injection
FastAPI leverages a dependency injection system (using `Depends`) to manage shared database connections, authentication credentials, and request filters.
For example, declaring a security dependency allows extracting auth headers like `X-API-Key` before executing route logic.

## Security Header Check
To secure routes in FastAPI, declare an API key header check. You can define a dependency function that reads the header value:
```python
def check_key(x_api_key: str = Header(...)):
    if x_api_key != "secret-token":
        raise HTTPException(status_code=401)
```
Using the `Security` wrapper ensures that authentication headers are verified and rate limit metrics are logged per request.
FastAPI handles CORS policies via CORSMiddleware, protecting the backend from unauthorized cross-origin requests.
