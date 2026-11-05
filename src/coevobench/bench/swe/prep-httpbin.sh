python -m pip install -q httpbin >/tmp/prep.log 2>&1 || pip install -q httpbin >>/tmp/prep.log 2>&1
grep -q "httpbin.org" /etc/hosts 2>/dev/null || echo "127.0.0.1 httpbin.org www.httpbin.org" >> /etc/hosts
openssl req -x509 -newkey rsa:2048 -keyout /tmp/hb.key -out /tmp/hb.crt -days 3650 -nodes \
  -subj "/CN=httpbin.org" -addext "subjectAltName=DNS:httpbin.org,DNS:www.httpbin.org,DNS:localhost,IP:127.0.0.1" >>/tmp/prep.log 2>&1
for CAB in "$(python -c 'import requests.certs;print(requests.certs.where())' 2>/dev/null)" \
           "$(python -c 'import certifi;print(certifi.where())' 2>/dev/null)"; do
  [ -n "$CAB" ] && [ -w "$CAB" ] && cat /tmp/hb.crt >> "$CAB"
done
python - >/tmp/httpbin.log 2>&1 <<'PYEOF' &
import ssl, threading, time, httpbin
from wsgiref.simple_server import make_server, WSGIServer, WSGIRequestHandler
from socketserver import ThreadingMixIn
class TS(ThreadingMixIn, WSGIServer):
    daemon_threads = True
class Q(WSGIRequestHandler):
    def log_message(self, *a):
        pass
def serve(port, tls):
    srv = make_server("0.0.0.0", port, httpbin.app, server_class=TS, handler_class=Q)
    if tls:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain("/tmp/hb.crt", "/tmp/hb.key")
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    srv.serve_forever()
threading.Thread(target=serve, args=(80, False), daemon=True).start()
threading.Thread(target=serve, args=(443, True), daemon=True).start()
open("/tmp/httpbin.url", "w").write("http://localhost/")
while True:
    time.sleep(3600)
PYEOF
for i in $(seq 1 60); do [ -s /tmp/httpbin.url ] && break; sleep 0.5; done
