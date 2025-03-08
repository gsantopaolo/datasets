from datasets import load_dataset

# Load the dataset
dataset = load_dataset("gsantopaolo/faa-balloon-flying-handbook")
print(dataset)

# Print the first 5 rows of the train split
for row in dataset["train"][:5]:
    print(row)
