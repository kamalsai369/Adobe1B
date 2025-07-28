import os
import json
from datetime import datetime

# === 🚨 FIXED INFO FOR THIS TEST CASE ==========

challenge_id = "round_1b_001"
test_case_name = "Travel Planner"
description = "France Travel"

persona_role = "Travel Planner"
job_task = "Plan a trip of 4 days for a group of 10 college friends."

# ===============================================

# Get current script path
base_path = os.path.dirname(os.path.abspath(__file__))
pdf_path = os.path.join(base_path, "pdf")

# Ensure 'pdf' folder exists
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"📂 'pdf/' folder not found at {pdf_path}")

# Load PDF filenames in sorted order
documents = []
for filename in sorted(os.listdir(pdf_path)):
    if filename.lower().endswith(".pdf"):
        title = os.path.splitext(filename)[0]
        documents.append({
            "filename": filename,
            "title": title
        })

if not documents:
    raise ValueError("❌ No PDFs found in pdf/ folder.")

# Compose final JSON
input_json = {
    "challenge_info": {
        "challenge_id": challenge_id,
        "test_case_name": test_case_name,
        "description": description
    },
    "documents": documents,
    "persona": {
        "role": persona_role
    },
    "job_to_be_done": {
        "task": job_task
    }
}

# Write to challenge1b_input.json
output_file = os.path.join(base_path, "challenge1b_input.json")
with open(output_file, "w") as f:
    json.dump(input_json, f, indent=4)

print(f"✅ challenge1b_input.json generated with {len(documents)} documents.")
