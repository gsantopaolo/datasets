import os
import glob
from typing import List
import argparse
from langchain_core.documents import Document
from string import Template
import os
import openai
from openai import OpenAI
import pandas as pd
from typing import Any, Dict
import json
import re

# Define a template with a placeholder for the manual text
request_template = Template(f"""Please generate 5 question and answer pairs based on the FAA Balloon Flight Manual text provided 
below. Each question should be crafted so that the answer is fully self-contained and understandable without needing 
any extra context. Ensure that the answers convey the meaning of the text without merely repeating it verbatim. Vary 
the style and format of the questions by incorporating some challenging or nuanced ones. Additionally, in at least 
one answer, reverse the order of some words compared to their sequence in the text to add creative variation.

Format your response in JSON array, where each element is an object with the keys 'question' and 'answer'.

Here is an example of your output: {{ "question": "What is the National Airspace System (NAS) as described in the FAA 
Balloon Flight Manual?", "answer": "The NAS is a comprehensive network that encompasses every element associated with 
U.S. airspace, including air navigation facilities, equipment, services, airports or landing areas, aeronautical 
charts, and essential regulatory and technical components." }}

Do not include any numbering for the questions.
Your ONLY response with a JSON object. 

Below is the FAA Balloon Flight Manual text for reference:

[FAA Balloon Flight Manual text Start]
$context
[FAA Balloon Flight Manual text End]

If the extracted text is insufficiently informative—such as when it consists solely of a page index or any fragment 
that cannot yield a meaningful Q&A pair—return nothing.""")

# Create an empty DataFrame with the required columns
columns = ['id', 'question', 'context', 'answer']
df: pd.DataFrame = pd.DataFrame(columns=columns)


def add_row(question: str, context: str, answer: str) -> None:
    """
    Add a new row to the DataFrame with an auto-incrementing id.

    Args:
        df (pd.DataFrame): The DataFrame to which the row is added.
        question (str): The question text.
        context (str): The context text.
        answer (str): The answer text.
    """
    new_id: int = len(df) + 1  # Auto-incrementing id
    new_row: dict[str, Any] = {
        'id': new_id,
        'question': question,
        'context': context,
        'answer': answer
    }
    # Append the new row; using .loc avoids deprecation warnings
    df.loc[len(df)] = new_row


def serialize_to_jsonl(file_path: str) -> None:
    """
    Serialize the DataFrame to a JSON Lines (JSONL) file.

    Args:
        df (pd.DataFrame): The DataFrame to serialize.
        file_path (str): The output file path.
    """
    df.to_json(file_path, orient="records", lines=True)


def generate_qa(context: str) -> List[Dict[str, str]]:
    """
    Generates 5 question and answer pairs from the given FAA Balloon Flight Manual text
    by calling the OpenAI API with a filled prompt template.

    Args:
        context (str): The FAA Balloon Flight Manual text to use in the prompt.

        Returns:
        List[Dict[str, str]]: A list of Q&A pairs as dictionaries.
    """
    # Skip processing if context is too short
    if len(context) < 100:
        print("Context too short (less than 100 characters). Skipping QA generation.")
        return []

    # Fill in the request template with the provided context
    prompt = request_template.substitute(context=context)
    print("*" * 80)
    # print(f"calling OpenAI API with prompt:\n {prompt}: \n")
    print(f"calling OpenAI API with context:\n {context}: \n")
    # Load OpenAI API token from environment variables
    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # Call the OpenAI API using the ChatCompletion endpoint.
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        # setting temperature=0.7 aims for a balance between creativity and reliability,
        # giving  responses that are neither too monotonous nor too erratic
        temperature=0.7,
    )

    content = response.choices[0].message.content

    print(f"response: \n: {content}")
    print("*" * 80)

    # Remove Markdown code block formatting if present.
    if content.startswith("```"):
        # Remove the first line (e.g. "```json") and the last line if it contains triple backticks.
        lines = content.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()

    # At this point, content should be a valid JSON array (e.g., "[]" or "[{...}, {...}, ...]")
    try:
        qa_list = json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON from API response: {e}")

    return qa_list


def clean_text(text: str) -> str:
    """
    Remove non-printable (noisy) characters from the text.

    Args:
        text (str): The original text.

    Returns:
        str: Cleaned text with only printable characters.
    """
    return ''.join(ch for ch in text if ch.isprintable())

def normalize_context(text: str) -> str:
    # Remove any 5-digit sequences (e.g., "18990", "19002", "19145")
    text = re.sub(r'\d{5}', '', text)
    # Optionally remove stray underscores that might be left behind
    text = text.replace('_', '')
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_by_delimiter(text: str, delimiter: str = "**") -> List[str]:
    """
    Split the text using a given multi-character delimiter.

    Args:
        text (str): The text to split.
        delimiter (str): The delimiter to split the text on.

    Returns:
        List[str]: A list of non-empty, stripped text chunks.
    """
    # Split by the delimiter and filter out any empty parts.
    parts = text.split(delimiter)
    return [part.strip() for part in parts if part.strip()]

def process_md_file(file_path: str, delimiter: str = "**") -> List[Document]:
    """
    Read a markdown file, clean its content, and split it into Document chunks using the delimiter.

    Args:
        file_path (str): Path to the Markdown file.
        delimiter (str): Delimiter used to split the content.

    Returns:
        List[Document]: List of Document objects containing the split chunks.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    cleaned_content = normalize_context(clean_text(content))
    # Split content by the specified delimiter
    chunks = split_by_delimiter(cleaned_content, delimiter)
    print(f"divided into {len(chunks)} chunks")
    # Convert each chunk into a Document; metadata can be extended as needed.
    return [Document(page_content=chunk, metadata={}) for chunk in chunks]

def process_folder(folder_path: str, delimiter: str = "**") -> None:
    """
    Process all Markdown files in a folder: clean and split them using the delimiter, then print results.

    Args:
        folder_path (str): Directory containing Markdown files.
        delimiter (str): Delimiter for splitting the file content.
    """
    md_files = glob.glob(os.path.join(folder_path, "*.md"))
    for file_path in md_files:
        print("*" * 80)
        print("*" * 80)
        print(f"Processing file: {file_path}")
        documents = process_md_file(file_path, delimiter)
        for i, doc in enumerate(documents):
            qa_list = generate_qa(doc.page_content)

            # Validate that qa_list is not empty and contains the required keys.
            if not qa_list:
                print(f"Skipping document {i + 1} due to empty QA response: {qa_list}")
                continue

            for qa in qa_list:
                if not qa.get("question") or not qa.get("answer"):
                    print(f"Skipping QA pair in document {i + 1} due to missing keys: {qa}")
                    continue

                add_row(
                    question=qa["question"],
                    context=doc.page_content,  # Fixed typo: page_content, not page_conten
                    answer=qa["answer"]
                )
            # print(f"\n--- Document {i + 1} ---")
            # print("Content:\n", doc.page_content)
            # print("-" * 40)

        print("*" * 80)
        print("*" * 80)


def main() -> None:
    """
    Main function to parse command-line arguments and process the folder.
    """
    parser = argparse.ArgumentParser(
        description="Split all Markdown files in a folder using '**' as the delimiter."
    )
    parser.add_argument("folder", type=str, help="Path to the folder containing Markdown files.")
    args = parser.parse_args()
    process_folder(args.folder, delimiter="**")

    # Serialize the DataFrame to a JSONL file
    serialize_to_jsonl("dataset.jsonl")
    print("Dataset saved as 'dataset.jsonl'.")

    # Read the JSONL file into a DataFrame
    # lines=True tells Pandas to treat each line as a separate JSON object
    df1 = pd.read_json('dataset.jsonl', lines=True)
    print("Number of rows:", df1.shape[0])

    grouped_df = df1.groupby('context').size().reset_index(name='count')
    print("Number of unique 'context' rows:", grouped_df.shape[0])

if __name__ == "__main__":
    main()

# usage  python3 split_md.py test/
# todo add the api call to gpt
# todo convert the retorning call in a jsonl file
# todo see what to do with eval
