import os
import polars as pl

def list_files_with_polars(directory, output_dir):
    test_files = []
    train_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            normalized_path = os.path.normpath(file_path)
            parts = normalized_path.split(os.sep)

            if "test" in parts:
                test_files.append({"file_path": file_path})
            elif "train" in parts:
                train_files.append({"file_path": file_path})

    # Create Polars DataFrames
    test_df = pl.DataFrame(test_files)
    train_df = pl.DataFrame(train_files)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Write to CSV
    test_df.write_csv(os.path.join(output_dir, "test.csv"))
    train_df.write_csv(os.path.join(output_dir, "train.csv"))

    return test_df, train_df


test_df, train_df = list_files_with_polars("chest_xray", "output_csvs")