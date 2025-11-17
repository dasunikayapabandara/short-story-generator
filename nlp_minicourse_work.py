# =======================================================
# NLP Activity – FULL PDF IMPLEMENTATION (CPU-only)
# Student: [YOUR NAME]
# Date: November 16, 2025
# =======================================================

from transformers import pipeline, set_seed

def cpu_pipe(task, model):
    return pipeline(task, model=model, device=-1)   # <-- CPU

print("=== PDF Activity – All Parts ===\n")

# ------------------- Part 1 (install) -------------------
print("Part 1: pip install transformers torch  [DONE]")

# ------------------- Part 2 -------------------
print("\n=== Part 2: Basic Generation ===")
gen = cpu_pipe("text-generation", "gpt2")
prompt = "In the future, artificial intelligence will"
set_seed(42)
out = gen(prompt, max_length=60, num_return_sequences=1, truncation=True)
print(out[0]["generated_text"])

# ------------------- Part 3 -------------------
print("\n=== Part 3: Controlled Generation ===")
prompt = "Education in the digital age requires"
out = gen(prompt, max_length=80, num_return_sequences=3,
          temperature=0.8, top_k=50, top_p=0.95, truncation=True)
for i, s in enumerate(out, 1):
    print(f"Sample {i}:\n{s['generated_text']}\n")

# ------------------- Part 4 -------------------
print("\n=== Part 4: Model Comparison ===")
for m in ["gpt2", "distilgpt2"]:
    g = cpu_pipe("text-generation", m)
    txt = g("Climate change is one of the most critical issues because",
            max_length=60, num_return_sequences=1, truncation=True)[0]["generated_text"]
    print(f"{m}:\n{txt}\n")

# ------------------- Part 5 -------------------
print("\n=== Part 5: Creative Texts ===")
cr = ["Success comes when you...", "Breaking News: Artificial Intelligence has..."]
for i, p in enumerate(cr, 1):
    set_seed(42)
    txt = gen(p, max_length=70, num_return_sequences=1, truncation=True)[0]["generated_text"]
    print(f"{i}. {txt}")
print("\nBest: #1 – realistic & motivational")

# ------------------- Part 6 (Poems) -------------------
print("\n=== Part 6: Prompt Sensitivity ===")
poems = [
    "Write a poem about artificial intelligence.",
    "As a poet, describe how AI transforms human emotions.",
    "Explain the impact of AI in a poetic way."
]
for p in poems:
    set_seed(42)
    print(f"\nPrompt: {p}")
    print(gen(p, max_length=80, num_return_sequences=1, truncation=True)[0]["generated_text"])

# ------------------- Part 6 (Summarization) -------------------
print("\n=== Part 6: Summarization ===")
summ = cpu_pipe("summarization", "facebook/bart-large-cnn")
text = """Artificial Intelligence is transforming education by enabling personalized learning 
paths, automated grading, and real-time feedback. However, ethical and privacy concerns 
remain significant challenges."""
s = summ(text, max_length=40, min_length=20, do_sample=False)
print("Summary:", s[0]['summary_text'])

# ------------------- Mini Project -------------------
print("\n" + "="*60)
print("CREATIVE MINI PROJECT – Short Story Generator")
print("="*60)
set_seed(42)
story = gen("Once upon a time in a futuristic city…",
            max_length=350, temperature=0.9, top_p=0.92,
            num_return_sequences=1, truncation=True)[0]["generated_text"]
print(story)

print("\nALL PDF PARTS COMPLETED SUCCESSFULLY!")
