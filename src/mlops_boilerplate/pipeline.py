
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from datetime import datetime

class DataProcessor:
    """Handles data loading, preprocessing, and splitting."""
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_data(self) -> pd.DataFrame:
        print(f"Loading data from {self.data_path}")
        # Simulate loading data from a CSV or similar source
        # For demonstration, create a dummy DataFrame
        data = {
            'feature_1': np.random.rand(100),
            'feature_2': np.random.rand(100) * 10,
            'feature_3': np.random.randint(0, 5, 100),
            'target': np.random.randint(0, 2, 100)
        }
        df = pd.DataFrame(data)
        return df

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Preprocessing data...")
        # Simulate some preprocessing steps
        df['feature_1_sq'] = df['feature_1'] ** 2
        return df

    def split_data(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
        print("Splitting data into training and testing sets...")
        X = df.drop(columns=[target_column])
        y = df[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

class ModelTrainer:
    """Trains and evaluates a machine learning model."""
    def __init__(self, model_name: str = "RandomForestClassifier"):
        self.model_name = model_name
        self.model = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        print(f"Training {self.model_name}...")
        self.model = RandomForestClassifier(random_state=42, **kwargs)
        self.model.fit(X_train, y_train)
        print("Model training complete.")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not trained yet.")
        print("Evaluating model...")
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        print(f"Model Accuracy: {accuracy:.4f}")
        return {"accuracy": accuracy, "report": report}

    def save_model(self, path: str):
        if self.model is None:
            raise ValueError("Model not trained yet.")
        print(f"Saving model to {path}")
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        print(f"Loading model from {path}")
        self.model = joblib.load(path)

class MLOpsPipeline:
    """Orchestrates the end-to-end MLOps workflow."""
    def __init__(self, data_path: str, model_dir: str = "models", metrics_dir: str = "metrics"):
        self.data_path = data_path
        self.model_dir = model_dir
        self.metrics_dir = metrics_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)

    def run(self):
        print("Starting MLOps Pipeline...")
        processor = DataProcessor(self.data_path)
        df = processor.load_data()
        df_processed = processor.preprocess_data(df)
        X_train, X_test, y_train, y_test = processor.split_data(df_processed, target_column='target')

        trainer = ModelTrainer()
        trainer.train(X_train, y_train, n_estimators=100, max_depth=10)
        metrics = trainer.evaluate(X_test, y_test)

        model_filename = os.path.join(self.model_dir, f"model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib")
        trainer.save_model(model_filename)

        metrics_filename = os.path.join(self.metrics_dir, f"metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json")
        with open(metrics_filename, "w") as f:
            json.dump(metrics, f, indent=4)
        
        print("MLOps Pipeline finished successfully.")
        print(f"Model saved to: {model_filename}")
        print(f"Metrics saved to: {metrics_filename}")

if __name__ == "__main__":
    # Create a dummy data file for the pipeline
    dummy_data_path = "dummy_data.csv"
    data = {
        'feature_1': np.random.rand(100),
        'feature_2': np.random.rand(100) * 10,
        'feature_3': np.random.randint(0, 5, 100),
        'target': np.random.randint(0, 2, 100)
    }
    pd.DataFrame(data).to_csv(dummy_data_path, index=False)

    pipeline = MLOpsPipeline(data_path=dummy_data_path)
    pipeline.run()
    os.remove(dummy_data_path) # Clean up dummy data

# Update on 2023-01-04 00:00:00
# Update on 2023-01-04 00:00:00
# Update on 2023-01-05 00:00:00
# Update on 2023-01-06 00:00:00
# Update on 2023-01-09 00:00:00
# Update on 2023-01-09 00:00:00
# Update on 2023-01-13 00:00:00
# Update on 2023-01-13 00:00:00
# Update on 2023-01-17 00:00:00
# Update on 2023-01-18 00:00:00
# Update on 2023-01-19 00:00:00
# Update on 2023-01-20 00:00:00
# Update on 2023-01-23 00:00:00
# Update on 2023-01-23 00:00:00
# Update on 2023-01-24 00:00:00
# Update on 2023-01-25 00:00:00
# Update on 2023-01-27 00:00:00
# Update on 2023-02-02 00:00:00
# Update on 2023-02-03 00:00:00
# Update on 2023-02-03 00:00:00
# Update on 2023-02-06 00:00:00
# Update on 2023-02-07 00:00:00
# Update on 2023-02-07 00:00:00
# Update on 2023-02-07 00:00:00
# Update on 2023-02-08 00:00:00
# Update on 2023-02-08 00:00:00
# Update on 2023-02-09 00:00:00
# Update on 2023-02-10 00:00:00
# Update on 2023-02-13 00:00:00
# Update on 2023-02-13 00:00:00
# Update on 2023-02-14 00:00:00
# Update on 2023-02-15 00:00:00
# Update on 2023-02-16 00:00:00
# Update on 2023-02-20 00:00:00
# Update on 2023-02-20 00:00:00
# Update on 2023-02-23 00:00:00
# Update on 2023-02-24 00:00:00
# Update on 2023-02-24 00:00:00
# Update on 2023-02-27 00:00:00
# Update on 2023-03-03 00:00:00
# Update on 2023-03-03 00:00:00
# Update on 2023-03-07 00:00:00
# Update on 2023-03-08 00:00:00
# Update on 2023-03-09 00:00:00
# Update on 2023-03-14 00:00:00
# Update on 2023-03-17 00:00:00