# Balloon Dataset

The **Balloon** dataset is derived from the  
[FAA Balloon Flying Handbook (FAA-H-8083-11B)](https://www.faa.gov/regulations_policies/handbooks_manuals/aviation/media/FAA-H-8083-11B.pdf). It was created by processing the official FAA manual to fly a hot air balloon. 


## Process Overview

### 1. PDF Conversion to Markdown
- **Downloaded Manuals**: The FAA manuals were first downloaded in PDF format.
- **Conversion Tool**: The PDFs were converted to Markdown using the `pymupdf4llm` tool. This conversion helps in leveraging Markdown’s structure for subsequent processing.

### 2. Splitting the Markdown
- **Analysis of Markdown Structure**: An analysis of the converted Markdown revealed that the document, being an older PDF, was best split using the bold markers (`**`).
- **Splitting Strategy**: The Markdown file was split into chunks based on these bold sections. This ensures that each chunk represents a coherent section of the manual.

### 3. Reusable Prompt Creation
- **Prompt Design**: A prompt was created that can be reused for other use cases. The prompt instructs a language model to generate 5 question-answer (QA) pairs for each text chunk.
- **Flexibility**: This prompt design allows for easy adaptation to different documents or further dataset creation tasks.

### 4. Generating Question-Answer Pairs
- **LLM Utilization**: Each Markdown chunk was sent to a language model with the reusable prompt.
- **QA Pair Creation**: The language model generated 5 QA pairs per chunk, effectively summarizing and questioning the content.

### 5. Aggregation into a DataFrame
- **Data Collection**: All the generated QA pairs, along with their corresponding context (the text chunks), were aggregated into a single DataFrame.
- **Purpose**: This aggregation facilitates easy manipulation and analysis of the dataset.

### 6. Conversion to JSONL Format
- **Why JSONL?**: The DataFrame was converted to JSONL (JSON Lines) format. JSONL is a convenient format where each line is a valid JSON object. 
This format is particularly useful for:
  - **Streaming Data**: Easy to process large datasets line by line.
  - **Machine Learning Workflows**: Supported by many tools and libraries.
- **Usage**: The JSONL file is ready for use in various ML pipelines and applications.

### 7. Usage with HuggingFace
The dataset is fully compatible with the HuggingFace `datasets` library. Here’s an example of how to load the dataset:

```python
from datasets import load_dataset

dataset = load_dataset('json', data_files='path/to/your/dataset.jsonl', split='train')
print(dataset[0])
```

## Conclusion

The **Balloon** dataset provides a structured set of question-answer pairs. By following the steps above—from PDF conversion to the final JSONL output—you now have a reusable pipeline that can be applied to similar documents. 
This dataset is ideal for tasks in natural language processing, such as fine-tuning LLMs for question-answering , document comprehension, and more.

