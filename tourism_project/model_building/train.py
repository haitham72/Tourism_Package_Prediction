import pandas as pd
import xgboost as xgb
import joblib
import mlflow
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, StandardScaler

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("MLOps_Tourism")

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze()
ytest = pd.read_csv("ytest.csv").squeeze()

numeric_features = Xtrain.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = Xtrain.select_dtypes(include=["object"]).columns.tolist()

# Ratio of negative to positive class handles the class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

param_grid = {
    "xgbclassifier__n_estimators": [50, 75, 100],
    "xgbclassifier__max_depth": [2, 3],
    "xgbclassifier__colsample_bytree": [0.4, 0.6],
    "xgbclassifier__colsample_bylevel": [0.4, 0.6],
    "xgbclassifier__learning_rate": [0.01, 0.1],
    "xgbclassifier__reg_lambda": [0.4, 0.6],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log every parameter combination as a nested run
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_score", results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", results["std_test_score"][i])

    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)

    classification_threshold = 0.45

    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= classification_threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= classification_threshold).astype(int)

    print("Train classification report:")
    print(classification_report(ytrain, y_pred_train))
    print("Test classification report:")
    print(classification_report(ytest, y_pred_test))

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1-score": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1-score": test_report["1"]["f1-score"]
    })

    model_path = "tourism_project/deployment/best_prediction.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")
