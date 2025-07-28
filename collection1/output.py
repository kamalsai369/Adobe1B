import os
import json
import re
import time
import datetime
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
LLM_MODEL = 'tinyllama'  # Less than 1GB model
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
INPUT_JSON_PATH = "challenge1b_input.json"
PDF_FOLDER_PATH = "pdf"
OUTPUT_JSON_PATH = "challenge1b_output.json"
TOP_K_SECTIONS = 5
MIN_WORDS = 80  # Ensure refined text has at least 80 words

# --- PDF Processing Functions ---

def extract_pdf_text_with_page(pdf_path):
    """Extracts text from each page of a PDF."""
    try:
        doc = fitz.open(pdf_path)
        page_texts = []
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                page_texts.append((page_num + 1, text))
        return page_texts
    except Exception as e:
        print(f"Error reading PDF {pdf_path}: {e}")
        return []

def extract_meaningful_title(text, max_length=100):
    """Extract a meaningful title from text."""
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if not lines:
        return "Untitled Section"
    
    # Try to find section headers by looking at formatting patterns
    for line in lines[:10]:  # Look at first 10 lines
        # Check if line looks like a title (short, capitalized, no period at end)
        if 2 < len(line.split()) < 8 and line[0].isupper() and not line.endswith('.'):
            return line[:max_length]
            
    # Fall back to the first substantive line
    for line in lines[:5]:
        if len(line) > 10:
            return line[:max_length]
            
    return lines[0][:max_length]

def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def extract_section_content(text, section_title):
    """Try to extract content related to a specific section title."""
    # This function attempts to find the text for a specific section
    lines = text.split('\n')
    start_idx = -1
    end_idx = -1
    
    # Find the section start
    for i, line in enumerate(lines):
        if section_title in line:
            start_idx = i
            break
    
    if start_idx == -1:
        return ""  # Section not found
        
    # Find the next section or take until the end
    for i in range(start_idx + 1, len(lines)):
        if i + 1 < len(lines) and lines[i].strip() and lines[i][0].isupper() and len(lines[i].split()) < 8:
            end_idx = i
            break
    
    if end_idx == -1:
        end_idx = len(lines)
        
    section_text = '\n'.join(lines[start_idx:end_idx])
    return section_text

def clean_text(text):
    """Removes unwanted special characters from text, keeping basic punctuation."""
    # This regex removes characters that are not alphanumeric, whitespace, or basic punctuation.
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    # Replace multiple spaces with a single space for cleaner output.
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Main Application Logic ---

def main():
    start_time = time.time()
    
    # 1. Load Input JSON
    try:
        with open(INPUT_JSON_PATH, "r", encoding="utf-8") as f:
            input_data = json.load(f)
        input_documents = [doc["filename"] for doc in input_data["documents"]]
        persona = input_data["persona"]["role"] 
        job_to_be_done = input_data["job_to_be_done"]["task"]
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    # 2. Process all PDF documents
    print("Step 1: Processing PDF documents...")
    all_chunks = []
    all_metadatas = []
    
    for pdf_file in input_documents:
        file_path = os.path.join(PDF_FOLDER_PATH, pdf_file)
        if not os.path.exists(file_path):
            print(f"Warning: File not found {file_path}")
            continue
            
        page_texts = extract_pdf_text_with_page(file_path)
        for page_number, text in page_texts:
            # Store both small chunks (for embedding search) and the full page (for context)
            chunks = chunk_text(text)
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": pdf_file,
                    "page_number": page_number,
                    "page_content_full": text,  # Store full page text for better content extraction
                    "section_title": extract_meaningful_title(text if i == 0 else chunk),
                    "chunk_id": i
                })

    if not all_chunks:
        print("Error: No text could be extracted from documents")
        return

    # 3. Create embeddings and find relevant sections
    print(f"Step 2: Creating embeddings with {EMBEDDING_MODEL}...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    doc_embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)
    query_embedding = embedding_model.encode(job_to_be_done)
    
    # Calculate similarity
    similarities = np.dot(doc_embeddings, query_embedding)
    top_indices = np.argsort(similarities)[::-1]
    
    # Ensure diversity by picking from different documents/pages
    retrieved_docs = []
    seen_sources = set()
    
    for idx in top_indices:
        if len(retrieved_docs) >= TOP_K_SECTIONS:
            break
            
        metadata = all_metadatas[idx]
        source_key = f"{metadata['source']}-{metadata['page_number']}"
        
        # Prioritize unique document-page combinations
        if source_key not in seen_sources:
            # Use the full page text for better context
            retrieved_docs.append({
                "content": metadata["page_content_full"],
                "metadata": metadata
            })
            seen_sources.add(source_key)

    # 4. Generate analysis with LLM
    print(f"Step 3: Generating analysis with LLM {LLM_MODEL}...")
    llm = Ollama(model=LLM_MODEL, temperature=0.7)  # Slightly higher temperature for more detail
    
    # Prompt designed to generate longer, more detailed responses
    prompt = PromptTemplate.from_template(
        """
        You are a {persona} helping with: "{job_to_be_done}"
        
        Analyze the following text and write a detailed summary (at least 100 words) that addresses the task.
        Include specific information like names, places, and recommendations from the text.
        Be comprehensive and informative.
        
        Text to analyze:
        "{content}"
        
        Your detailed analysis (minimum 100 words):
        """
    )
    chain = prompt | llm | StrOutputParser()
    
    extracted_sections = []
    subsection_analysis = []
    
    # Process each relevant document
    for rank, doc in enumerate(retrieved_docs, 1):
        metadata = doc["metadata"]
        content = doc["content"]
        section_title = metadata["section_title"]
        
        # Try to extract more focused content for the specific section
        section_content = extract_section_content(content, section_title) or content
        
        # Generate refined text with explicit minimum length requirement
        refined_text = chain.invoke({
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "content": section_content
        })
        
        # Ensure minimum length - if too short, try again with stronger instruction
        word_count = len(refined_text.split())
        if word_count < MIN_WORDS:
            retry_prompt = PromptTemplate.from_template(
                """
                You are a {persona} helping with: "{job_to_be_done}"
                
                I need a DETAILED analysis of at least 100 words (this is required) for the following text.
                Include specific details, examples, and recommendations from the text.
                
                Text to analyze:
                "{content}"
                
                Your DETAILED analysis (MUST be at least 100 words):
                """
            )
            retry_chain = retry_prompt | llm | StrOutputParser()
            refined_text = retry_chain.invoke({
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "content": section_content
            })
        
        # Clean the generated text before adding it to the output
        cleaned_text = clean_text(refined_text)

        # Add to output
        extracted_sections.append({
            "document": metadata["source"],
            "section_title": section_title,
            "importance_rank": rank,
            "page_number": metadata["page_number"]
        })
        
        subsection_analysis.append({
            "document": metadata["source"],
            "refined_text": cleaned_text,
            "page_number": metadata["page_number"]
        })

    # 5. Write output
    output_data = {
        "metadata": {
            "input_documents": input_documents,
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }
    
    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
        
    end_time = time.time()
    print(f"✅ Success! Processing completed.")
    print(f"Output saved to {OUTPUT_JSON_PATH}")

if __name__ == "__main__":
    main()