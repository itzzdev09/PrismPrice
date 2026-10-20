"""Small ASGI security controls for the public HTTP boundary."""

from __future__ import annotations

from starlette.types import ASGIApp, Message, Receive, Scope, Send

__all__ = ["RequestBodyLimitMiddleware", "SecurityHeadersMiddleware"]


class RequestBodyLimitMiddleware:
    """Reject request bodies larger than the configured limit before parsing."""

    def __init__(self, app: ASGIApp, max_bytes: int = 256 * 1024) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return

        content_length = next(
            (
                value
                for name, value in scope.get("headers", [])
                if name.lower() == b"content-length"
            ),
            None,
        )
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                await self._respond(scope, send, 400, b'{"detail":"Invalid Content-Length"}')
                return
            if declared_length < 0:
                await self._respond(scope, send, 400, b'{"detail":"Invalid Content-Length"}')
                return
            if declared_length > self.max_bytes:
                await self._respond(scope, send, 413, b'{"detail":"Request body too large"}')
                return

        chunks: list[bytes] = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            if message["type"] != "http.request":
                continue
            body = message.get("body", b"")
            size += len(body)
            if size > self.max_bytes:
                await self._respond(scope, send, 413, b'{"detail":"Request body too large"}')
                return
            chunks.append(body)
            if not message.get("more_body", False):
                break

        request_body = b"".join(chunks)
        body_sent = False

        async def replay_body() -> Message:
            nonlocal body_sent
            if not body_sent:
                body_sent = True
                return {"type": "http.request", "body": request_body, "more_body": False}
            return await receive()

        await self.app(scope, replay_body, send)

    @staticmethod
    async def _respond(scope: Scope, send: Send, status: int, body: bytes) -> None:
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ]
        if scope.get("http_version") in {"1.0", "1.1"}:
            headers.append((b"connection", b"close"))
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": headers,
            }
        )
        await send({"type": "http.response.body", "body": body})


class SecurityHeadersMiddleware:
    """Set browser security headers on every response."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        async def add_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                headers.extend(
                    [
                        (b"x-content-type-options", b"nosniff"),
                        (b"x-frame-options", b"DENY"),
                        (b"referrer-policy", b"strict-origin-when-cross-origin"),
                        (
                            b"content-security-policy",
                            b"default-src 'self'; base-uri 'self'; object-src 'none'; "
                            b"frame-ancestors 'none'; form-action 'self'; img-src 'self' data:; "
                            b"style-src 'self' 'unsafe-inline'; script-src 'self'",
                        ),
                        (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
                    ]
                )
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, add_headers)
