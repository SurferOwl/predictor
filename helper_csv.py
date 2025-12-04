import pandas as pd
import os

def save_unique_rows(input_csv_path, output_csv_path=None):
    """
    Reads a CSV file, removes duplicate rows, and saves the unique rows to a new CSV.
    
    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str, optional): Path to save the CSV with unique rows.
            If None, saves in the same directory as input with '_unique' suffix.
    
    Returns:
        int: Number of unique rows saved.
    """
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"File not found: {input_csv_path}")

    # Load CSV
    df = pd.read_csv(input_csv_path)
    
    # Drop duplicate rows
    unique_df = df.drop_duplicates()
    
    # Determine output path
    if output_csv_path is None:
        base, ext = os.path.splitext(input_csv_path)
        output_csv_path = f"{base}_unique{ext}"
    
    # Save unique rows to CSV
    unique_df.to_csv(output_csv_path, index=False)
    
    print(f"Saved {len(unique_df)} unique rows to: {output_csv_path}")
    return len(unique_df)


save_unique_rows("C:\\Licenta\\predictor\\data\\Final_Augmented_dataset_Diseases_and_Symptoms.csv","C:\\Licenta\\predictor\\data\\Final_Augmented_dataset_Diseases_and_Symptoms_Revised.csv")
