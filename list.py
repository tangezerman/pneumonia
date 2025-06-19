import os
import polars as pl

def list_files_with_polars(directory):
    test_files = []
    train_files = []
    val_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == ".DS_Store":
                continue

            file_path = os.path.join(root, file)
            normalized_path = os.path.normpath(file_path)
            parts = normalized_path.split(os.sep)

            if "test" in parts:
                test_files.append({"file_path": file_path})
            elif "train" in parts:
                train_files.append({"file_path": file_path})
            elif "val" in parts:
                val_files.append({"file_path": file_path})

    # Create Polars DataFrames
    test_df = pl.DataFrame(test_files)
    train_df = pl.DataFrame(train_files)
    val_df = pl.DataFrame(val_files)

    # Ensure output directory exists
    # Write to CS
    test_df.write_csv("test.csv")
    train_df.write_csv("train.csv")
    val_df.write_csv("val.csv")

    return test_df, train_df, val_df


def combine_df(df1_path, df2_path, df3_path):
    df1 = pl.read_csv(df1_path)
    df2 = pl.read_csv(df2_path)
    df3 = pl.read_csv(df3_path)

    # Get the maximum length to pad shorter column with None
    max_len = max(len(df1), len(df2), len(df3))

    # Create padded series
    test_series = df1["file_path"].to_list()
    train_series = df2["file_path"].to_list()
    val_series = df3["file_path"].to_list()

    # Pad with None values to match max length
    test_series.extend([None] * (max_len - len(test_series)))
    train_series.extend([None] * (max_len - len(train_series)))
    val_series.extend([None] * (max_len - len(val_series)))

    # Create new dataframe with both columns
    test_padded = pl.DataFrame({"test": test_series})
    train_padded = pl.DataFrame({"train": train_series})
    val_padded = pl.DataFrame({"val": val_series})

    # Combine the columns horizontally
    combined_df = pl.concat(
        [train_padded, test_padded, val_padded], how="horizontal")

    # Remove None entries
    combined_df = combined_df.filter(
        (pl.col("train").is_not_null()) |
        (pl.col("test").is_not_null()) |
        (pl.col("val").is_not_null())
    )

    # Write to CSV
    combined_df.write_csv("dataset.csv")


test_df, train_df, val_df = list_files_with_polars("chest_xray")

combine_df("test.csv", "train.csv", "val.csv")
