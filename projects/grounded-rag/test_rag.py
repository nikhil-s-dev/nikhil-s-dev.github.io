import json
import unittest
from pathlib import Path
from rag import Index


class RagTests(unittest.TestCase):
    def setUp(self):
        self.index = Index(json.loads(Path(__file__).with_name('documents.json').read_text()))

    def test_restricted_document_filtered_before_ranking(self):
        public = self.index.search('fictional pump R17 vibration limit', ['public'])
        self.assertNotIn('restricted-01', [d['id'] for d in public])
        maintenance = self.index.search('fictional pump R17 vibration limit', ['maintenance'])
        self.assertEqual(maintenance[0]['id'], 'restricted-01')

    def test_answer_is_evidence_and_has_citation(self):
        result = self.index.answer('permit energy isolation', ['public'])
        self.assertIn('energy isolation', result['answer'])
        self.assertEqual(result['citations'][0]['id'], 'permit-01')

    def test_no_evidence_and_duplicate_ids(self):
        self.assertEqual(self.index.answer('xylophone', ['public'])['citations'], [])
        with self.assertRaises(ValueError): Index([{'id': 'a'}, {'id': 'a'}])


if __name__ == '__main__': unittest.main()
