# Validation

The trusted course dataset loaded successfully: 13,348 rows, 10 columns and 1024-dimensional embeddings. The archived selected model is an MLPClassifier. Three source-row predictions completed after correcting the duplicated AI-label feature block (64 columns were being produced where the saved model expects 16). The standalone predictor and both notebook copies now align to the same saved schema. Three regression cases cover all four AI labels, missing AI labels, and a partial/unseen label. Full model comparison and retraining were not run. The model metadata records scikit-learn 1.7.0; requirements pin that version.

Preparation used Python 3.11 in an existing research environment. These checks are not a clean-environment, end-to-end replication of every result.
