# Hour 7 Research: Security & Authentication Patterns

## Overview
Research on security and authentication patterns for Shadow AI, based on GitHub projects.

---

## FastAPI (96.8k stars)
**GitHub**: https://github.com/fastapi/fastapi

### Key Security Features
- Built-in OAuth2 with JWT token support
- HTTP Basic authentication
- APIKey authentication
- Dependency injection for auth
- Type-safe validation

### Pattern
```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from jose import JWTError, jwt
from datetime import datetime, timedelta

# OAuth2 Bearer Token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# JWT Token Creation
def create_access_token(data: dict, expires_delta: timedelta):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Token Verification
async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    return username
```

---

## OAuthLib (3k stars)
**GitHub**: https://github.com/oauthlib/oauthlib

### Key Concepts
- OAuth 1.0 (RFC 5849)
- OAuth 2.0 (RFC 6749)
- Framework-agnostic implementation
- OpenID Connect support
- JWT Bearer tokens

### Pattern (OAuth2 Authorization Code Flow)
```python
from oauthlib.oauth2 import WebApplicationClient

client = WebApplicationClient(client_id)

# Authorization URL
authorization_url = client.prepare_request_uri(
    authorization_uri,
    redirect_uri=redirect_uri,
    scope=['read', 'write']
)

# Token Request
token_url, headers, body = client.prepare_token_request(
    token_uri,
    authorization_response=redirect_uri,
    redirect_url=redirect_uri,
    code=code
)

# Token Response
token = client.parse_request_body_response(token_response)
```

---

## Full Stack FastAPI Template (42.5k stars)
**GitHub**: https://github.com/fastapi/full-stack-fastapi-template

### Key Security Features
- JWT authentication with refresh tokens
- Password hashing with bcrypt
- Email-based password recovery
- API key authentication
- Rate limiting
- CORS configuration

### Pattern
```python
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class Security:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    def get_password_hash(self, password: str) -> str:
        return pwd_context.hash(password)
    
    def create_access_token(self, subject: str, expires_delta: Optional[timedelta] = None) -> str:
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        
        to_encode = {"exp": expire, "sub": str(subject)}
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, subject: str) -> str:
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode = {"exp": expire, "sub": str(subject), "type": "refresh"}
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
```

---

## HTTPX (15.1k stars)
**GitHub**: https://github.com/encode/httpx

### Security Features
- HTTP/2 support with ALPN
- Strict timeouts
- SSL/TLS verification
- Basic/Digest authentication
- Cookie persistence

### Pattern
```python
import httpx

# SSL Verification
client = httpx.Client(verify=True)

# Basic Auth
auth = httpx.BasicAuth(username="user", password="secret")

# Timeout Configuration
timeout = httpx.Timeout(10.0, connect=5.0, read=30.0, write=10.0)

# HTTPS Client
async with httpx.AsyncClient(timeout=timeout, verify=True) as client:
    response = await client.get("https://api.example.com", auth=auth)
```

---

## Key Security Patterns for Shadow AI

### 1. JWT Authentication System
```python
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import jwt, JWTError
from passlib.context import CryptContext
import secrets

class JWTAuth:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        
        self.access_token_expire = 30  # minutes
        self.refresh_token_expire = 7   # days
        self.api_key_length = 32
    
    def create_access_token(self, user_id: str, scopes: list = None, 
                           expires_delta: timedelta = None) -> str:
        """Create JWT access token."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire)
        
        payload = {
            "sub": user_id,
            "exp": expire,
            "type": "access",
            "scopes": scopes or [],
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token."""
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire)
        payload = {
            "sub": user_id,
            "exp": expire,
            "type": "refresh",
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            raise ValueError("Invalid token")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return self.pwd_context.hash(password)
```

### 2. API Key Authentication
```python
from dataclasses import dataclass
from typing import Optional
import secrets
import hashlib

@dataclass
class APIKey:
    key_id: str
    key_hash: str
    user_id: str
    scopes: list
    expires_at: Optional[datetime]
    created_at: datetime

class APIKeyManager:
    def __init__(self):
        self.keys = {}  # In production, use database
    
    def generate_key(self, user_id: str, scopes: list = None, prefix: str = "sk") -> str:
        """Generate new API key."""
        raw_key = f"{prefix}_{secrets.token_urlsafe(32)}"
        key_id = hashlib.sha256(raw_key.encode()).hexdigest()[:16]
        key_hash = hashlib.sha512(raw_key.encode()).hexdigest()
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            scopes=scopes or ["read", "write"],
            expires_at=datetime.utcnow() + timedelta(days=365),
            created_at=datetime.utcnow()
        )
        
        self.keys[key_id] = api_key
        return raw_key  # Only time raw key is shown
    
    def verify_key(self, raw_key: str) -> Optional[APIKey]:
        """Verify API key."""
        key_hash = hashlib.sha512(raw_key.encode()).hexdigest()
        
        for key in self.keys.values():
            if key.key_hash == key_hash:
                if key.expires_at and datetime.utcnow() > key.expires_at:
                    return None
                return key
        return None
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke API key."""
        if key_id in self.keys:
            del self.keys[key_id]
            return True
        return False
```

### 3. Rate Limiting
```python
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

class RateLimiter:
    def __init__(self, requests_per_minute: int = 60, 
                 requests_per_hour: int = 1000):
        self.rate_per_minute = requests_per_minute
        self.rate_per_hour = requests_per_hour
        self.requests = defaultdict(list)
        self.lock = asyncio.Lock()
    
    async def is_allowed(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        async with self.lock:
            now = datetime.utcnow()
            
            # Clean old requests
            self.requests[client_id] = [
                t for t in self.requests[client_id]
                if now - t < timedelta(hours=1)
            ]
            
            minute_requests = len([
                t for t in self.requests[client_id]
                if now - t < timedelta(minutes=1)
            ])
            
            hour_requests = len(self.requests[client_id])
            
            if minute_requests >= self.rate_per_minute:
                return False
            
            if hour_requests >= self.rate_per_hour:
                return False
            
            self.requests[client_id].append(now)
            return True
    
    def get_remaining(self, client_id: str) -> dict:
        """Get remaining requests."""
        now = datetime.utcnow()
        minute_requests = len([
            t for t in self.requests[client_id]
            if now - t < timedelta(minutes=1)
        ])
        hour_requests = len([
            t for t in self.requests[client_id]
            if now - t < timedelta(hours=1)
        ])
        
        return {
            "minute_remaining": self.rate_per_minute - minute_requests,
            "hour_remaining": self.rate_per_hour - hour_requests
        }
```

### 4. Input Validation
```python
from pydantic import BaseModel, Field, validator, EmailStr
from typing import Optional, List
import re

class UserCreate(BaseModel):
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50, 
                          regex=r"^[a-zA-Z0-9_-]+$")
    password: str = Field(..., min_length=8, max_length=100)
    
    @validator('password')
    def password_strength(cls, v):
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain lowercase letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain digit")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Password must contain special character")
        return v

class APIRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000)
    context: Optional[dict] = None
    scopes: List[str] = Field(default_factory=list)
    
    @validator('scopes')
    def validate_scopes(cls, v):
        allowed = {"read", "write", "admin"}
        if not set(v).issubset(allowed):
            raise ValueError(f"Invalid scopes. Allowed: {allowed}")
        return v
```

### 5. Security Middleware
```python
from fastapi import Request, HTTPException, status
from typing import Callable

class SecurityMiddleware:
    def __init__(self, auth_manager, rate_limiter):
        self.auth_manager = auth_manager
        self.rate_limiter = rate_limiter
        self.excluded_paths = {"/health", "/metrics", "/docs", "/openapi.json"}
    
    async def __call__(self, request: Request, call_next: Callable):
        # Skip excluded paths
        if request.url.path in self.excluded_paths:
            return await call_next(request)
        
        # Rate limiting
        client_id = self._get_client_id(request)
        if not await self.rate_limiter.is_allowed(client_id):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        # Authentication
        auth_header = request.headers.get("Authorization")
        if auth_header:
            token = auth_header.replace("Bearer ", "")
            try:
                payload = self.auth_manager.decode_token(token)
                request.state.user = payload
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
        
        # Security headers
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """Get client identifier for rate limiting."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0]
        return request.client.host
```

---

## Summary
- **FastAPI**: Built-in OAuth2/JWT support, dependency injection for auth
- **OAuthLib**: Framework-agnostic OAuth1/OAuth2 implementation
- **Full Stack Template**: Complete auth with JWT, password hashing, refresh tokens
- **HTTPX**: Secure HTTP client with SSL/TLS verification
- **Key patterns**: JWT tokens, API keys, rate limiting, input validation, security headers