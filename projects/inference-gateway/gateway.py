"""Small local gateway for an OpenAI-compatible inference server."""
import argparse
import http.client
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

MAX_BODY = 1024 * 1024


def make_handler(upstream, limit, timeout):
    target = urlsplit(upstream.rstrip('/'))
    if target.scheme not in ('http', 'https') or not target.hostname:
        raise ValueError('upstream must be an http(s) URL')
    slots = threading.BoundedSemaphore(limit)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Do not write prompts or authorization headers to logs.

        def respond(self, code, message):
            body = json.dumps({'error': message}).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != '/health':
                return self.respond(404, 'not found')
            body = b'{"status":"ok"}'
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.path != '/v1/chat/completions':
                return self.respond(404, 'not found')
            try:
                length = int(self.headers.get('Content-Length', ''))
            except ValueError:
                return self.respond(411, 'valid content length required')
            if length <= 0 or length > MAX_BODY:
                return self.respond(413, 'request size exceeds limit')
            if not slots.acquire(blocking=False):
                return self.respond(429, 'gateway concurrency limit reached')
            conn = None
            sent = False
            try:
                body = self.rfile.read(length)
                json.loads(body)
                factory = http.client.HTTPSConnection if target.scheme == 'https' else http.client.HTTPConnection
                conn = factory(target.hostname, target.port, timeout=timeout)
                headers = {'Content-Type': 'application/json', 'Accept': self.headers.get('Accept', 'application/json')}
                # Configure a service credential; never accept a client-supplied upstream key.
                if os.environ.get('UPSTREAM_API_KEY'):
                    headers['Authorization'] = 'Bearer ' + os.environ['UPSTREAM_API_KEY']
                conn.request('POST', target.path.rstrip('/') + '/v1/chat/completions', body, headers)
                result = conn.getresponse()
                self.send_response(result.status)
                self.send_header('Content-Type', result.getheader('Content-Type', 'application/json'))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                sent = True
                # HTTP/1.0 connection close frames the response; read1 avoids buffering the full stream.
                while chunk := result.read1(4096):
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (json.JSONDecodeError, UnicodeDecodeError):
                if not sent:
                    self.respond(400, 'invalid JSON')
            except (OSError, http.client.HTTPException):
                if not sent:
                    self.respond(502, 'upstream unavailable')
            finally:
                if conn:
                    conn.close()
                slots.release()

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upstream', default='http://127.0.0.1:8000')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--max-inflight', type=int, default=8)
    parser.add_argument('--timeout', type=float, default=60)
    args = parser.parse_args()
    if args.max_inflight < 1 or args.timeout <= 0:
        parser.error('max-inflight and timeout must be positive')
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(args.upstream, args.max_inflight, args.timeout))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
