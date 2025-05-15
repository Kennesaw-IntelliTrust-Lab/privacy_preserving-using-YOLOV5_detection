import json

# Load outputs from full_train.json
with open("/mnt/bst/hxu10/hxu10/chanti/llama_rag/data/full_train.json", "r") as f:
    data = json.load(f)

# Clean and format context chunks from "output"
clean_chunks = []
for i, item in enumerate(data):
    answer = item.get("output", "").strip()
    if len(answer) > 30 and "I don't have enough information" not in answer:
        clean_chunks.append(answer)

# Save to rag_docs.txt
rag_file = "/mnt/bst/hxu10/hxu10/chanti/llama_rag/data/rag_docs.txt"
with open(rag_file, "w") as f:
    for chunk in clean_chunks:
        f.write(chunk.strip() + "\n\n")

print(f"✅ {len(clean_chunks)} context chunks saved to {rag_file}")
