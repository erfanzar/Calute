# WebSocket TLS test fixture

`websocket-localhost-cert.pem` and `websocket-localhost-key.pem` are a deliberately
self-signed `localhost` certificate/key pair used only by the offline WebSocket
monitor TLS rejection test. They are public test fixtures and must never be used
for authentication, production TLS, or any external service.
