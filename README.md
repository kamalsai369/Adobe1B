<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <meta name="description" content="A lightweight PDF document intelligence system with persona-driven analysis using RAG and TinyLLaMA." />
  <meta name="author" content="Your Name or Team Name" />

</head>
<body>

<h1>Document Intelligence System - Persona-Driven Analysis</h1>

<p>
  A dynamic document intelligence system that extracts and prioritizes relevant sections from PDF collections based on specific personas and their job-to-be-done tasks. This system works across diverse domains including academic research, business analysis, educational content, and more.
</p>

<h2>Project Overview</h2>
<p>This system implements a Retrieval-Augmented Generation (RAG) pipeline that:</p>
<ul>
  <li>Processes any collection of PDF documents (3–10 files)</li>
  <li>Adapts to different personas (researchers, analysts, students, etc.)</li>
  <li>Extracts relevant content based on specific tasks</li>
  <li>Generates structured analysis with importance ranking</li>
</ul>

<h2>Project Structure</h2>
<pre><code>collection1/
├── challenge1b_input.json    # Generated input configuration
├── challenge1b_output.json   # Generated analysis results
├── input.py                  # Input JSON generator
├── output.py                 # Main processing engine
└── pdf/                      # PDF document collection
    ├── document1.pdf
    ├── document2.pdf
    └── ...
</code></pre>

<h2>System Requirements</h2>
<ul>
  <li><strong>CPU-only execution</strong> (no GPU required)</li>
  <li><strong>Model size ≤ 1GB</strong> constraint</li>
  <li><strong>Processing time ≤ 60 seconds</strong> for 3–5 documents</li>
  <li><strong>No internet access</strong> during execution</li>
</ul>

<h2>Dependencies</h2>
<p>Install required Python packages:</p>
<pre><code>pip install PyMuPDF sentence-transformers numpy langchain-community</code></pre>

<p>Install Ollama for local LLM:</p>
<ul>
  <li>Follow <a href="https://github.com/ollama/ollama" target="_blank">Ollama installation guide</a></li>
  <li>Pull the lightweight model: Run this command---><code>ollama pull tinyllama</code></li>
</ul>

<h2>Usage Instructions</h2>

<h3>Step 1: Prepare Your Documents</h3>
<pre><code>pdf/
├── research_paper1.pdf
├── research_paper2.pdf
├── annual_report_2023.pdf
├── chemistry_chapter1.pdf
└── ...
</code></pre>

<h3>Step 2: Configure Test Case</h3>
<pre><code># === 🚨 FIXED INFO FOR THIS TEST CASE ==========
challenge_id = "round_1b_001"
test_case_name = "Academic Research"
description = "Literature Review"

persona_role = "PhD Researcher"
job_task = "Prepare comprehensive literature review"
</code></pre>

<h3>Step 3: Generate Input Configuration</h3>
<pre><code>cd collection1
python input.py</code></pre>

<h3>Step 4: Process Documents</h3>
<pre><code>python output.py</code></pre>

<h2>Example Test Cases</h2>

<h3>Academic Research</h3>
<pre><code>persona_role = "PhD Researcher in Computational Biology"
job_task = "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"</code></pre>

<h3>Business Analysis</h3>
<pre><code>persona_role = "Investment Analyst"
job_task = "Analyze revenue trends, R&D investments, and market positioning strategies"</code></pre>

<h3>Educational Content</h3>
<pre><code>persona_role = "Undergraduate Chemistry Student"
job_task = "Identify key concepts and mechanisms for exam preparation on reaction kinetics"</code></pre>

<h3>Travel Planning (Current Test Case)</h3>
<pre><code>persona_role = "Travel Planner"
job_task = "Plan a trip of 4 days for a group of 10 college friends."</code></pre>

<h2>Output Format</h2>
<pre><code>{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "Your Persona Role",
    "job_to_be_done": "Your Specific Task",
    "processing_timestamp": "2025-07-28T10:16:18.771443+00:00"
  },
  "extracted_sections": [
    {
      "document": "source_document.pdf",
      "section_title": "Relevant Section Header",
      "importance_rank": 1,
      "page_number": 5
    }
  ],
  "subsection_analysis": [
    {
      "document": "source_document.pdf",
      "refined_text": "Detailed analysis (minimum 80 words) addressing the persona's task...",
      "page_number": 5
    }
  ]
}
</code></pre>

<h2>Technical Implementation</h2>

<h3>Core Components</h3>
<ol>
  <li><strong>Document Processing</strong>: PyMuPDF extracts text with page-level granularity</li>
  <li><strong>Semantic Search</strong>: sentence-transformers (<code>all-MiniLM-L6-v2</code>) creates embeddings</li>
  <li><strong>Content Retrieval</strong>: Cosine similarity finds most relevant sections</li>
  <li><strong>Analysis Generation</strong>: tinyllama LLM produces persona-specific insights</li>
  <li><strong>Diversity Filtering</strong>: Ensures sections come from different documents/pages</li>
</ol>

<h2>Customization</h2>

<h3>Adding New Test Cases</h3>
<ol>
  <li>Place new PDFs in <code>collection1/pdf/</code></li>
  <li>Update <code>persona_role</code> and <code>job_task</code> in <code>input.py</code></li>
  <li>Re-run <code>input.py</code> and <code>output.py</code></li>
</ol>

<h3>Modifying Analysis Depth</h3>
<pre><code>TOP_K_SECTIONS = 5
MIN_WORDS = 80
chunk_size = 500</code></pre>

<h3>Changing the LLM Model</h3>
<pre><code>LLM_MODEL = "your_model_name"  # Ensure size ≤ 1GB</code></pre>

<h2>Performance Metrics</h2>
<ul>
  <li><strong>Processing Speed</strong>: &lt;60 seconds for 3–5 documents</li>
  <li><strong>Model Size</strong>: tinyllama (~637MB)</li>
  <li><strong>Accuracy</strong>: Semantic search ensures relevance</li>
  <li><strong>Coverage</strong>: Extracts from diverse sources</li>
</ul>

<h2>Troubleshooting</h2>

<h3>Common Issues</h3>
<div class="warning">
<ol>
  <li><strong>No PDFs found</strong>: Check <code>collection1/pdf/</code></li>
  <li><strong>Ollama not running</strong>: Start with <code>ollama serve</code></li>
  <li><strong>Model not found</strong>: Run <code>ollama pull tinyllama</code></li>
  <li><strong>Short analysis</strong>: Auto-retries until 80-word threshold met</li>
</ol>
</div>

<h3>Performance Optimization</h3>
<ul>
  <li>Use SSD storage</li>
  <li>Ensure 4GB+ RAM</li>
  <li>Close unused applications during execution</li>
</ul>

<h2>Quick Start Example</h2>

<div class="success">
<pre><code># 1. Install dependencies
pip install PyMuPDF sentence-transformers numpy langchain-community
ollama pull tinyllama

# 2. Place PDFs in collection1/pdf/

# 3. Set persona and task in input.py

# 4. Run processing
cd collection1
python input.py
python output.py

# 5. Check output in challenge1b_output.json
</code></pre>
</div>

<p><strong>This system provides a robust, scalable solution for document intelligence across any domain while maintaining strict performance and resource constraints.</strong></p>

</body>
</html>
