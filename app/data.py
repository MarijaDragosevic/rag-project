from langchain.document_loaders import GutenbergLoader
import re

# Load the document from Gutenberg
loader = GutenbergLoader("https://www.gutenberg.org/cache/epub/100/pg100.txt")
document = loader.load()


document_content = document[0].page_content
output_file_path = "data/complete_works_of_Shakespeare.txt"

def optimize_newlines(text: str) -> str:
   
    sections = re.split(r'\n{2,}', text)
    cleaned_sections = []
    
    for section in sections:
        cleaned_section = " ".join(section.splitlines()).strip()
        cleaned_sections.append(cleaned_section)
    return "\n".join(cleaned_sections)


document_content=optimize_newlines(document_content)

# Save the document content into the data folder
with open(output_file_path, "w", encoding="utf-8") as file:
    file.write(document_content)

print(f"Document saved to {output_file_path}")
