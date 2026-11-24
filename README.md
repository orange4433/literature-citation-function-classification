# A Multi-Classification Recognition Model for Literature Citation Functions

Applied multivariate-statistics coursework on **citation-type classification**, combining text embeddings with AI-generated citation labels and annotator-credibility filtering.

## Report and implementation

- [Complete course report (HTML)](paper/course_report.html): original rendered report, preserved in its original language.
- `report_final.ipynb`: data inspection, credibility analysis, feature engineering, annotator-separated train/test split, model comparison and model export. Notebook outputs are cleared; the rendered report preserves the original results.
- `pku_labels_train.pkl`: the supplied course dataset, including 1024-dimensional embeddings and replacement annotation-group identifiers. See `PUBLIC_DATA_NOTES.md`.
- `saved_models/`: the archived selected model, label encoder, scaler, AI-feature schema and inference script.

The task is supervised citation-label classification. The original report compares multiple classifiers, including tree ensembles, SVM, neural networks and gradient boosting.

## Running

Use Python 3.11 and install `requirements.txt`. Run the notebook from the repository root so that the dataset and `saved_models/` paths resolve correctly. Full model comparison can take considerable time.

For inference, import `predict_new_data` from `saved_models/citation_predictor.py`, or run the script from the repository root with a trusted input pickle/CSV. Input rows require `Para_text`, `Ref_Clean_citation` and `embedding_vector`; optional AI-label columns must match the saved feature schema. A serialized list in CSV is not automatically parsed into a numeric embedding; the notebook's pickle schema is the supported example.

## Reproducibility

The dataset and saved model components are included. Dataset-schema and saved-model inference checks are recorded in `VALIDATION.md`. Full model training and model ranking were not rerun during archive preparation; binary model compatibility depends on the scikit-learn version used to save it.

This public copy replaces original record and annotator identifiers and removes email addresses from citation text. The rendered report uses replacement group identifiers. The original data and original report remain in a separate private archive. See [public-data notes](PUBLIC_DATA_NOTES.md) for the effects on reproduction. Load pickle/joblib files only from trusted sources.

README and source comments are in English; the original report, feature names and runtime messages are preserved.
