import unittest
import os
import json
from ingest import load_data, aggregate_daily
from model import train_models
from logger import log_prediction
from app import app

class TestAAVAILPipeline(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_01_ingest(self):
        df = load_data()
        self.assertFalse(df.empty)
        daily = aggregate_daily(df, country="United Kingdom")
        self.assertIn("revenue", daily.columns)

    def test_02_model_training(self):
        df = aggregate_daily(load_data())
        metrics = train_models(df, test_mode=True)
        self.assertTrue(os.path.exists(metrics["model_path"]))
        self.assertIn("baseline_rmse", metrics)

    def test_03_logging_isolation(self):
        log_prediction([100]*7, 500.0, 0.01, country="EIRE", test_mode=True)
        log_file = os.path.join("logs", "test_api.log")
        self.assertTrue(os.path.exists(log_file))

    def test_04_api_endpoint(self):
        payload = {
            "country": "Germany",
            "input": [200, 250, 300, 280, 290, 310, 330],
            "test_mode": True
        }
        response = self.app.post("/predict", data=json.dumps(payload), content_type="application/json")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["country"], "Germany")
        self.assertIn("prediction", data)

if __name__ == "__main__":
    unittest.main()