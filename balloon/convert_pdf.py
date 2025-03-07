#!/usr/bin/env python3

import os
from typing import List, Optional
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter


def convert_pdf_to_md(pdf_path: str, md_path: str) -> None:
    """
    Converts a single PDF file into Markdown and writes it to md_path.

    :param pdf_path: The file path to the PDF document.
    :param md_path: The file path where the resulting Markdown should be written.
    """
    markdown_output = pymupdf4llm.to_markdown(pdf_path)

    with open(md_path, "w", encoding="utf-8") as md_file:
        md_file.write(markdown_output)


def convert_folder_pdfs_to_md(folder_path: str) -> List[str]:
    """
    Searches through the specified folder for PDF files, converts each to Markdown,
    and saves them with the same base name (but .md extension).

    :param folder_path: Path to the folder containing PDF files.
    :return: A list of converted Markdown file paths for reference or logging.
    """
    converted_files = []
    # Ensure the folder exists
    if not os.path.isdir(folder_path):
        raise ValueError(f"The path '{folder_path}' is not a valid directory.")

    for entry in os.listdir(folder_path):
        if entry.lower().endswith(".pdf"):
            pdf_file_path = os.path.join(folder_path, entry)
            md_file_name = f"{os.path.splitext(entry)[0]}.md"
            md_file_path = os.path.join(folder_path, md_file_name)

            convert_pdf_to_md(pdf_file_path, md_file_path)
            converted_files.append(md_file_path)

    return converted_files


def main(folder_path: str) -> None:
    """
    Main entry point of the script. Converts all PDFs in the given folder to Markdown.

    :param folder_path: Path to the folder containing PDF files to be converted.
    """
    converted_list = convert_folder_pdfs_to_md(folder_path)
    if converted_list:
        print("Conversion complete! The following Markdown files were created:")
        for md_file in converted_list:
            print(f"  - {md_file}")
    else:
        print("No PDF files found in the specified folder.")


if __name__ == "__main__":
    # Example usage:
    # Adjust the folder_path to point to your PDF folder.
    folder_path_example = "docs/"
    main(folder_path_example)
