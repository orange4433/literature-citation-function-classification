"""Regression checks for the saved model's AI-label feature layout."""
from pathlib import Path
import importlib.util
import unittest
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "predictor", ROOT / "saved_models" / "citation_predictor.py"
)
predictor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predictor)


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.components = predictor.load_model_components(str(ROOT / "saved_models"))

    def check_prediction(self, labels):
        row = {"Para_text": "Example paragraph", "Ref_Clean_citation": "[1]",
               "embedding_vector": np.zeros(1024)}
        row.update(labels)
        result = predictor.predict_new_data(pd.DataFrame([row]), self.components)
        self.assertEqual(len(result), 1)
        self.assertIn(result.iloc[0]["预测标签"], self.components["label_encoder"].classes_)
        self.assertTrue(np.isfinite(result.iloc[0]["预测概率"]))

    def test_all_four_ai_labels(self):
        self.check_prediction({label: "Background" for label in
                               self.components["ai_info"]["ai_labels"]})

    def test_missing_ai_labels(self):
        self.check_prediction({})

    def test_partial_and_unseen_ai_label(self):
        self.check_prediction({"Citation_Type_GPT": "Unseen category"})


if __name__ == "__main__":
    unittest.main()
