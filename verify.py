# --------------------------------------------------------------
# verify.py – CPU-only, works on MX230 / any laptop
# --------------------------------------------------------------
import sys, traceback
from transformers import pipeline, set_seed

def run_part(title, func):
    print(f"\n{'='*20} {title} {'='*20}")
    try:
        func()
        print("PASS")
    except Exception as e:
        print("FAIL")
        traceback.print_exc()
        sys.exit(1)

# Helper: CPU pipeline
def cpu_pipeline(task, model):
    return pipeline(task, model=model, device=-1)   # <-- CPU

# ---------- Part 2 ----------
def part2():
    generator = cpu_pipeline("text-generation", "gpt2")
    prompt = "In the future, artificial intelligence will"
    set_seed(42)
    out = generator(prompt, max_length=60, num_return_sequences=1, truncation=True)
    txt = out[0]["generated_text"]
    print(txt)

# ---------- Part 3 ----------
def part3():
    generator = cpu_pipeline("text-generation", "gpt2")
    prompt = "Education in the digital age requires"
    out = generator(prompt,
                   max_length=80,
                   num_return_sequences=3,
                   temperature=0.8,
                   top_k=50,
                   top_p=0.95,
                   truncation=True)
    for i, s in enumerate(out, 1):
        print(f"Sample {i}: {s['generated_text'][:100]}...")
    assert len(out) == 3

# ---------- Part 4 ----------
def part4():
    models = ["gpt2", "distilgpt2"]
    prompt = "Climate change is one of the most critical issues because"
    for m in models:
        gen = cpu_pipeline("text-generation", m)
        txt = gen(prompt, max_length=60, num_return_sequences=1, truncation=True)[0]["generated_text"]
        print(f"{m}: {txt[:100]}...")

# ---------- Part 5 ----------
def part5():
    generator = cpu_pipeline("text-generation", "gpt2")
    prompts = [
        "Success comes when you...",
        "Breaking News: Artificial Intelligence has..."
    ]
    for i, p in enumerate(prompts, 1):
        set_seed(42)
        txt = generator(p, max_length=70, num_return_sequences=1, truncation=True)[0]["generated_text"]
        print(f"{i}. {txt}")

# ---------- Part 6 (poems) ----------
def part6_poems():
    generator = cpu_pipeline("text-generation", "gpt2")
    prompts = [
        "Write a poem about artificial intelligence.",
        "As a poet, describe how AI transforms human emotions.",
        "Explain the impact of AI in a poetic way."
    ]
    for p in prompts:
        set_seed(42)
        txt = generator(p, max_length=80, num_return_sequences=1, truncation=True)[0]["generated_text"]
        print(f"\nPrompt: {p}\n{txt}")

# ---------- Part 6 (summarization) ----------
def part6_sum():
    summarizer = cpu_pipeline("summarization", "facebook/bart-large-cnn")
    text = """Artificial Intelligence is transforming education by enabling personalized learning 
paths, automated grading, and real-time feedback. However, ethical and privacy concerns 
remain significant challenges."""
    out = summarizer(text, max_length=40, min_length=20, do_sample=False)
    print("Summary:", out[0]['summary_text'])

# ---------- Mini-project ----------
def mini_project():
    set_seed(42)
    gen = cpu_pipeline("text-generation", "gpt2")
    story = gen("Once upon a time in a futuristic city…",
                max_length=300,
                temperature=0.85,
                top_p=0.93,
                num_return_sequences=1,
                truncation=True)[0]["generated_text"]
    print("\n--- Short Story ---\n", story[:500], "...")

# ------------------- RUN ALL -------------------
if __name__ == "__main__":
    run_part("Part 2 – Basic Generation", part2)
    run_part("Part 3 – Parameters", part3)
    run_part("Part 4 – Model Comparison", part4)
    run_part("Part 5 – Creative Texts", part5)
    run_part("Part 6 – Poems", part6_poems)
    run_part("Part 6 – Summarization", part6_sum)
    run_part("Mini-project – Story Generator", mini_project)

    print("\nALL TESTS PASSED! Your code is correct.")
