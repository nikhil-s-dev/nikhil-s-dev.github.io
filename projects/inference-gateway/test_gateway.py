import http.client
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from gateway import make_handler


class Backend(BaseHTTPRequestHandler):
    def log_message(self, *args): pass
    def do_POST(self):
        self.rfile.read(int(self.headers['Content-Length']))
        body = b'data: {"choices":[]}\n\ndata: [DONE]\n\n'
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class GatewayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.backend = ThreadingHTTPServer(('127.0.0.1', 0), Backend)
        cls.gateway = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(f'http://127.0.0.1:{cls.backend.server_port}', 2, 2))
        for server in (cls.backend, cls.gateway):
            threading.Thread(target=server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        for server in (cls.backend, cls.gateway):
            server.shutdown(); server.server_close()

    def request(self, method, path, body=None):
        conn = http.client.HTTPConnection('127.0.0.1', self.gateway.server_port, timeout=3)
        conn.request(method, path, body)
        response = conn.getresponse(); result = response.status, response.read(), response.getheader('Content-Type')
        conn.close(); return result

    def test_stream_and_content_type_are_forwarded(self):
        status, body, content_type = self.request('POST', '/v1/chat/completions', b'{"stream":true}')
        self.assertEqual(status, 200); self.assertIn(b'[DONE]', body); self.assertEqual(content_type, 'text/event-stream')

    def test_health_and_invalid_json(self):
        self.assertEqual(self.request('GET', '/health')[0], 200)
        self.assertEqual(self.request('POST', '/v1/chat/completions', b'{')[0], 400)
        self.assertEqual(self.request('GET', '/missing')[0], 404)

    def test_unavailable_upstream(self):
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler('http://127.0.0.1:1', 1, .2))
        threading.Thread(target=server.serve_forever, daemon=True).start()
        conn = http.client.HTTPConnection('127.0.0.1', server.server_port)
        try:
            conn.request('POST', '/v1/chat/completions', '{}'); self.assertEqual(conn.getresponse().status, 502)
        finally:
            conn.close(); server.shutdown(); server.server_close()


if __name__ == '__main__': unittest.main()
