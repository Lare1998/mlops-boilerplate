
from mlops_boilerplate.pipeline import MLOpsPipeline
import os
import pandas as pd
import numpy as np

if __name__ == "__main__":
    print("Starting MLOps Pipeline application...")

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

    print("MLOps Pipeline application finished.")

# Update on 2023-01-02 00:00:00
# Update on 2023-01-03 00:00:00
# Update on 2023-01-03 00:00:00
# Update on 2023-01-04 00:00:00
# Update on 2023-01-05 00:00:00
# Update on 2023-01-09 00:00:00