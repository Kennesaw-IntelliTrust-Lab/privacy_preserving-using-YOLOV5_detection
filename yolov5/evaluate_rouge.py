import json
import pandas as pd
from rouge_score import rouge_scorer

# ✅ Paths to your files
gt_path = "/mnt/bst/hxu10/hxu10/chanti/llama_rag/data/full_train.json"
pred_path = "/mnt/bst/hxu10/hxu10/chanti/llama_rag/data/llama_generated.json"
save_path = "/mnt/bst/hxu10/hxu10/chanti/llama_rag/data/rouge_evaluation.csv"

# ✅ Load ground truth and predictions
with open(gt_path, "r") as f:
    ground_truth = json.load(f)

with open(pred_path, "r") as f:
    predictions = json.load(f)

# ✅ Sanity check
assert len(ground_truth) == len(predictions), "Mismatch in number of samples!"

# ✅ Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# ✅ Store results
rows = []

for idx, (gt, pred) in enumerate(zip(ground_truth, predictions)):
    ref = gt["output"]
    hyp = pred["output"]
    score = scorer.score(ref, hyp)

    rows.append({
        "Instruction": gt["instruction"],
        "Reference": ref,
        "Generated": hyp,
        "ROUGE-1": score["rouge1"].fmeasure,
        "ROUGE-2": score["rouge2"].fmeasure,
        "ROUGE-L": score["rougeL"].fmeasure
    })

# ✅ Save to CSV
df = pd.DataFrame(rows)
df.to_csv(save_path, index=False)

# ✅ Show average scores
print("📊 ROUGE Evaluation Completed!")
print("Average ROUGE-1:", df["ROUGE-1"].mean())
print("Average ROUGE-2:", df["ROUGE-2"].mean())
print("Average ROUGE-L:", df["ROUGE-L"].mean())
print(f"✅ Scores saved to: {save_path}")
