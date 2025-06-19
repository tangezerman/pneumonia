import os
import polars as pl

def list_files_with_polars(directory):
    test_files = []
    train_files = []
    validation_files = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Skip .DS_Store files
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
                validation_files.append({"file_path": file_path})  # Fixed: was adding to train_files
    
    # Create Polars DataFrames
    test_df = pl.DataFrame(test_files)
    train_df = pl.DataFrame(train_files)
    validation_df = pl.DataFrame(validation_files)
    
    # Write to CSV
    test_df.write_csv("test.csv")
    train_df.write_csv("train.csv")
    validation_df.write_csv("validation.csv")
    
    return test_df, train_df, validation_df

def clean_dataframe(df):
    """
    Remove rows containing '.DS_Store' and any null entries from a DataFrame
    
    Args:
        df: Polars DataFrame to clean
        
    Returns:
        Cleaned Polars DataFrame
    """
    # Remove rows containing '.DS_Store' in any column
    df_cleaned = df.filter(~pl.any_horizontal(pl.all().str.contains(".DS_Store")))
    
    # Remove rows where all columns are null
    df_cleaned = df_cleaned.filter(~pl.all_horizontal(pl.all().is_null()))
    
    return df_cleaned

def combine_df(df1_path, df2_path, df3_path):
    df1 = pl.read_csv(df1_path)
    df2 = pl.read_csv(df2_path)
    df3 = pl.read_csv(df3_path)
    
    # Get the maximum length to pad shorter columns with None
    max_len = max(len(df1), len(df2), len(df3))
    
    # Create padded series
    test_series = df1["file_path"].to_list()
    train_series = df2["file_path"].to_list()
    validation_series = df3["file_path"].to_list()
    
    # Pad with None values to match max length
    test_series.extend([None] * (max_len - len(test_series)))
    train_series.extend([None] * (max_len - len(train_series)))
    validation_series.extend([None] * (max_len - len(validation_series)))
    
    # Create new dataframe with all three columns
    train_padded = pl.DataFrame({"train": train_series})
    test_padded = pl.DataFrame({"test": test_series})
    validation_padded = pl.DataFrame({"validation": validation_series})
    
    # Combine the columns horizontally
    combined_df = pl.concat([train_padded, test_padded, validation_padded], how="horizontal")
    
    # Clean the dataframe (remove .DS_Store and null entries)
    combined_df = clean_dataframe(combined_df)
    
    # Write to CSV
    combined_df.write_csv("dataset.csv")

test_df, train_df, validation_df = list_files_with_polars("chest_xray")

combine_df("train.csv", "test.csv", "validation.csv")