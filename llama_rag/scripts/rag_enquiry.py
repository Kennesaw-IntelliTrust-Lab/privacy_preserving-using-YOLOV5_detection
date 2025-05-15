import os
import json
import warnings
import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import login

# ----------------------
# Configurations
# ----------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Hugging Face Login
HF_TOKEN = "YOUR-HF TOKEN"
login(token=HF_TOKEN)

# ----------------------
# Load LLaMA-2 Model
# ----------------------
print("[INFO] Loading LLaMA-2 tokenizer and model...")
model_id = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    token=HF_TOKEN
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    num_beams=4,
    do_sample=False,
    top_p=1.0,
    repetition_penalty=1.2,
    pad_token_id=tokenizer.eos_token_id,
    return_full_text=False
)

llm = HuggingFacePipeline(pipeline=pipe)

# ----------------------
# Load Embeddings
# ----------------------
print("[INFO] Loading embedding model...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ----------------------
# Load or Build FAISS
# ----------------------
db_path = "/mnt/bst/hxu10/hxu10/chanti/llama_rag/scripts/rag_index"
print("[INFO] Checking FAISS index...")

if os.path.exists(db_path) and os.path.isdir(db_path):
    vectorstore = FAISS.load_local(db_path, embeddings=embedding_model, allow_dangerous_deserialization=True)
else:
    print("[WARN] Index not found. Building new FAISS index...")
    from langchain_core.documents import Document
    with open("/mnt/bst/hxu10/hxu10/chanti/llama_rag/data/rag_docs.txt", "r") as f:
        content = f.read()
    docs = [Document(page_content=p.strip()) for p in content.split("\n\n") if p.strip()]
    vectorstore = FAISS.from_documents(docs, embedding_model)
    os.makedirs(db_path, exist_ok=True)
    vectorstore.save_local(db_path)
    print("[INFO] FAISS index created and saved.")

# ----------------------
# Prompt Template
# ----------------------
prompt_template = """
<s>[INST]
You are an AI assistant specialized in plant disease information.

Context:
{context}

Question:
{input}

Respond with a clear and concise answer based only on the context. If the context lacks the answer, reply with "I don't have enough information."
[/INST]
"""
prompt = ChatPromptTemplate.from_template(prompt_template)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# ----------------------
# Clean Output
# ----------------------
def clean_output(text):
    if isinstance(text, list):
        text = text[0]
    bad_phrases = [
        "Thank you for asking!", "I'm here to help you with that.",
        "Let me know if you have any further questions.",
        "Based on the context, here is my answer:", "Of course!", "I hope this helps!"
    ]
    for phrase in bad_phrases:
        text = text.replace(phrase, "")
    return text.strip()

# ----------------------
# Run Inference
# ----------------------
input_json = "/mnt/bst/hxu10/hxu10/chanti/llama_rag/data/full_train.json"
output_json = "/mnt/bst/hxu10/hxu10/chanti/llama_rag/data/llama_generated.json"

print("[INFO] Loading input instructions...")
with open(input_json, "r") as f:
    ground_data = json.load(f)

llama_results = []
print("[INFO] Generating answers using LLaMA-2 + RAG...\n")

for idx, item in enumerate(ground_data):
    instruction = item["instruction"]
    print(f"[{idx + 1}/{len(ground_data)}] {instruction}")
    try:
        result = retrieval_chain.invoke({"input": instruction})
        cleaned_output = clean_output(result["answer"])
    except Exception as e:
        cleaned_output = f"Error: {str(e)}"
    llama_results.append({"instruction": instruction, "output": cleaned_output})

with open(output_json, "w") as f:
    json.dump(llama_results, f, indent=2)

print(f"[INFO] All results saved to: {output_json}")
