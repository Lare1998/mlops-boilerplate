
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
# Update on 2023-01-11 00:00:00
# Update on 2023-01-11 00:00:00
# Update on 2023-01-13 00:00:00
# Update on 2023-01-16 00:00:00
# Update on 2023-01-16 00:00:00
# Update on 2023-01-18 00:00:00
# Update on 2023-01-20 00:00:00
# Update on 2023-01-23 00:00:00
# Update on 2023-01-24 00:00:00
# Update on 2023-01-24 00:00:00
# Update on 2023-01-26 00:00:00
# Update on 2023-01-27 00:00:00
# Update on 2023-01-30 00:00:00
# Update on 2023-02-01 00:00:00
# Update on 2023-02-02 00:00:00
# Update on 2023-02-03 00:00:00
# Update on 2023-02-06 00:00:00
# Update on 2023-02-08 00:00:00
# Update on 2023-02-09 00:00:00
# Update on 2023-02-09 00:00:00
# Update on 2023-02-10 00:00:00
# Update on 2023-02-15 00:00:00
# Update on 2023-02-15 00:00:00
# Update on 2023-02-16 00:00:00
# Update on 2023-02-17 00:00:00
# Update on 2023-02-20 00:00:00
# Update on 2023-02-21 00:00:00
# Update on 2023-02-21 00:00:00
# Update on 2023-02-24 00:00:00
# Update on 2023-02-27 00:00:00
# Update on 2023-03-01 00:00:00
# Update on 2023-03-01 00:00:00
# Update on 2023-03-02 00:00:00
# Update on 2023-03-07 00:00:00
# Update on 2023-03-08 00:00:00
# Update on 2023-03-09 00:00:00
# Update on 2023-03-10 00:00:00
# Update on 2023-03-15 00:00:00
# Update on 2023-03-22 00:00:00
# Update on 2023-03-24 00:00:00
# Update on 2023-03-28 00:00:00
# Update on 2023-03-28 00:00:00
# Update on 2023-03-29 00:00:00
# Update on 2023-04-03 00:00:00
# Update on 2023-04-05 00:00:00
# Update on 2023-04-05 00:00:00
# Update on 2023-04-10 00:00:00
# Update on 2023-04-10 00:00:00
# Update on 2023-04-10 00:00:00
# Update on 2023-04-11 00:00:00
# Update on 2023-04-11 00:00:00
# Update on 2023-04-13 00:00:00
# Update on 2023-04-13 00:00:00
# Update on 2023-04-14 00:00:00
# Update on 2023-04-14 00:00:00
# Update on 2023-04-17 00:00:00
# Update on 2023-04-17 00:00:00
# Update on 2023-04-17 00:00:00
# Update on 2023-04-18 00:00:00
# Update on 2023-04-20 00:00:00