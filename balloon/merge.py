import pandas as pd
from sklearn.model_selection import train_test_split

def main() -> None:
    # Read the first dataset
    df1 = pd.read_json('dataset_1.jsonl', lines=True)
    print("Number of rows in dataset1:", df1.shape[0])

    # Read the second dataset
    df2 = pd.read_json('dataset_2.jsonl', lines=True)
    print("Number of rows in dataset2:", df2.shape[0])

    # Read the third dataset
    df3 = pd.read_json('dataset_3.jsonl', lines=True)
    print("Number of rows in dataset3:", df3.shape[0])

    # Concatenate the two datasets
    combined_df = pd.concat([df1, df2, df3], ignore_index=True)
    print("Total number of rows in combined dataset:", combined_df.shape[0])

    # Assign new unique IDs starting from 1
    combined_df['id'] = range(1, len(combined_df) + 1)

    # Split the combined dataset into training (80%) and temp (20%) sets
    train_df, temp_df = train_test_split(combined_df, test_size=0.2, random_state=42)
    print("Number of rows in training set:", train_df.shape[0])

    # Split the temp dataset into validation (10%) and test (10%) sets
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    print("Number of rows in validation set:", val_df.shape[0])
    print("Number of rows in test set:", test_df.shape[0])

    # Save the datasets to JSONL files
    train_df.to_json('train.jsonl', orient='records', lines=True)
    val_df.to_json('validation.jsonl', orient='records', lines=True)
    test_df.to_json('test.jsonl', orient='records', lines=True)
    print("Datasets saved as 'train.jsonl', 'validation.jsonl', and 'test.jsonl'.")

if __name__ == "__main__":
    main()
