"""Local, extractive RAG prototype using an access-filtered BM25 index."""
import json
import math
import re
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def tokens(text):
    return re.findall(r'[a-z0-9]+', text.lower())


class Index:
    def __init__(self, documents):
        ids = [doc['id'] for doc in documents]
        if len(ids) != len(set(ids)):
            raise ValueError('document IDs must be unique')
        self.documents = documents

    def search(self, query, groups, k=3):
        # Filter before ranking so neither content nor corpus statistics leak across groups.
        allowed = [d for d in self.documents if set(d['groups']).intersection(groups)]
        if not allowed or not tokens(query): return []
        docs = [tokens(d['text']) for d in allowed]
        avg = sum(map(len, docs)) / len(docs) or 1
        terms = set(tokens(query)); n = len(docs); scores = []
        for doc, words in zip(allowed, docs):
            counts = Counter(words); score = 0.0
            for term in terms:
                df = sum(term in words2 for words2 in docs)
                if not df or not counts[term]: continue
                idf = math.log(1 + (n - df + .5) / (df + .5))
                freq = counts[term]
                score += idf * freq * 2.2 / (freq + 1.2 * (.25 + .75 * len(words) / avg))
            if score > 0:
                scores.append({'id': doc['id'], 'title': doc['title'], 'text': doc['text'], 'score': round(score, 6)})
        return sorted(scores, key=lambda d: (-d['score'], d['id']))[:k]

    def answer(self, query, groups):
        hits = self.search(query, groups)
        if not hits:
            return {'answer': 'No matching evidence found.', 'citations': []}
        # Extractive: return the source verbatim rather than generating unsupported claims.
        return {'answer': hits[0]['text'], 'citations': [{'id': hits[0]['id'], 'title': hits[0]['title']}]}


def handler(index):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            if self.path != '/ask': return self.reply(404, {'error': 'not found'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= 16384: return self.reply(413, {'error': 'invalid body size'})
                body = json.loads(self.rfile.read(length))
                query, groups = body['query'], body['groups']
                if not isinstance(query, str) or not query.strip() or not isinstance(groups, list) or not all(isinstance(x, str) for x in groups):
                    raise ValueError('bad request')
                self.reply(200, index.answer(query, groups))
            except (ValueError, KeyError, TypeError):
                self.reply(400, {'error': 'expected query and groups'})
        def reply(self, code, data):
            payload = json.dumps(data).encode()
            self.send_response(code); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload))); self.end_headers(); self.wfile.write(payload)
    return Handler


if __name__ == '__main__':
    docs = json.loads(Path(__file__).with_name('documents.json').read_text())
    server = ThreadingHTTPServer(('127.0.0.1', 8090), handler(Index(docs)))
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()
