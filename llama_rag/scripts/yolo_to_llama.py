import os
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Function to run YOLOv5 detection
def run_yolov5_detection(image_path, model_weights):
    detect_script = "/mnt/bst/hxu10/hxu10/chanti/yolov5/detect.py"
    command = [
        "python", detect_script,
        "--weights", model_weights,
        "--source", image_path,
        "--save-txt",
        "--project", "/mnt/bst/hxu10/hxu10/chanti/yolov5/runs/detect",
        "--name", "rag_sync",
        "--exist-ok"
    ]
    subprocess.run(command)

    label_dir = "/mnt/bst/hxu10/hxu10/chanti/yolov5/runs/detect/rag_sync/labels"
    if not os.path.exists(label_dir):
        raise FileNotFoundError("❌ No detection found or label file missing.")

    label_files = [f for f in os.listdir(label_dir) if f.endswith(".txt")]
    if not label_files:
        raise FileNotFoundError("❌ Label file not found after detection.")

    label_path = os.path.join(label_dir, label_files[0])
    with open(label_path, "r") as f:
        first_line = f.readline().strip()
        class_id = int(first_line.split()[0])

    class_map = {0: "bacterial_leaf_blight", 1: "rice_blast", 2: "brown_spot"}
    return class_map.get(class_id, "unknown")

# Load LLaMA-2 model
def load_llama_model():
    model_id = "meta-llama/Llama-2-7b-chat-hf"
    HF_TOKEN = os.getenv("HF_TOKEN")  # Use environment variable for safety

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        token=HF_TOKEN
    )

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
    )

# Run LLaMA RAG pipeline
def run_rag_pipeline(detected_class, pipe):
    # Step 1: Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 2: Load FAISS index
    db_path = "/mnt/bst/hxu10/hxu10/chanti/llama_rag/scripts/rag_index"
    vectorstore = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

    # Step 3: Set up retriever
    retriever = vectorstore.as_retriever(search_kwargs={
        "k": 5,
        "fetch_k": 10,
        "score_threshold": 0.0
    })

    # Step 4: Prepare prompt template
    prompt_template = """
    <s>[INST] <<SYS>>
    You are an AI assistant specialized in answering questions about plant diseases.
    Your task is to provide accurate, clear, and concise answers based solely on the provided context.
    <</SYS>>

    The following is context information about plant diseases:
    ---
    {context}
    ---

    Given the context information and not prior knowledge, answer this question: {input}

    Remember:
    1. Only use facts stated in the context
    2. If the context doesn't contain the answer, say "I don't have enough information"
    3. Do not make up information
    4. Keep your answer concise and to the point
    5. Format your answer clearly [/INST]
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Step 5: Create the chain
    question = f"What is {detected_class.replace('_', ' ')} and how can it be treated?"

# Retrieve top k context documents
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs])

# Fill the prompt with context and question
    final_prompt = prompt_template.format(context=context, input=question)

# Generate the response using HF pipeline
    response = pipe(final_prompt)[0]["generated_text"]

    return response
# Main execution
if __name__ == "__main__":
    image_path = "/mnt/bst/hxu10/hxu10/chanti/dataset/images/test/blast_orig_004.png"
    model_weights = "/mnt/bst/hxu10/hxu10/chanti/yolov5/runs/train/exp/weights/best.pt"

    detected_class = run_yolov5_detection(image_path, model_weights)
    print(f"✅ Detected class: {detected_class}")

    pipe = load_llama_model()
    response = run_rag_pipeline(detected_class, pipe)
    print(f"\n🤖 LLaMA-2 Response:\n{response}")