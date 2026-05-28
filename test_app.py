import sys
import json
import unittest
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app, load_sentiment_model

class TestApp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure a trained model exists and is loaded
        load_sentiment_model(clf_name="logistic_regression", vec_name="tfidf", force_train=False)

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<!DOCTYPE html>', response.data)
        self.assertIn(b'Sentiment Analysis Dashboard', response.data)

    def test_api_status(self):
        response = self.app.get('/api/status')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'loaded')
        self.assertIn('classifier', data)
        self.assertIn('vectorizer', data)
        self.assertIn('vocabulary_size', data)

    def test_api_predict_success(self):
        payload = {"text": "I love this tool, it is amazing!"}
        response = self.app.post('/api/predict', 
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['text'], payload['text'])
        self.assertIn('label', data)
        self.assertIn(data['label'], ['positive', 'negative', 'neutral'])
        self.assertIn('confidence', data)
        self.assertIn('probabilities', data)

    def test_api_predict_empty_text(self):
        payload = {"text": ""}
        response = self.app.post('/api/predict', 
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_api_predict_batch_success(self):
        payload = {
            "texts": [
                "This is wonderful",
                "It is terrible",
                "It is average"
            ]
        }
        response = self.app.post('/api/predict_batch', 
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('results', data)
        self.assertEqual(len(data['results']), 3)
        self.assertIn('summary', data)
        self.assertEqual(data['summary']['positive'] + data['summary']['negative'] + data['summary']['neutral'], 3)

    def test_api_train_success(self):
        payload = {
            "classifier": "naive_bayes",
            "vectorizer": "count"
        }
        response = self.app.post('/api/train', 
                                 data=json.dumps(payload),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertEqual(data['classifier'], 'naive_bayes')
        self.assertEqual(data['vectorizer'], 'count')
        self.assertIn('metrics', data)
        self.assertIn('accuracy', data['metrics'])

if __name__ == '__main__':
    unittest.main(verbosity=2)
