# --------------------------------------------------------------
# Language Generation using Transformers – Full Activity + Mini Project
# --------------------------------------------------------------

# ---------- Part 1: Imports ----------
from transformers import pipeline, set_seed
import torch

# ---------- Part 2: Basic Text Generation ----------
print("\n=== Part 2: Basic Text Generation ===")
generator = pipeline("text-generation", model="gpt2")

prompt = "In the future, artificial intelligence will"
set_seed(42)                                   # reproducible results

output = generator(prompt,
                   max_length=60,
                   num_return_sequences=1)

print(output[0]["generated_text"])

# Try a few more prompts
extra_prompts = [
    "The best way to learn programming is",
    "Once upon a time in a galaxy far away",
    "Quantum computers will change the world by"
]
for p in extra_prompts:
    set_seed(42)
    print("\nPrompt:", p)
    print(generator(p, max_length=60, num_return_sequences=1)[0]["generated_text"])


# ---------- Part 3: Controlling Generation Parameters ----------
print("\n=== Part 3: Controlling Parameters ===")
prompt = "Education in the digital age requires"

output = generator(prompt,
                   max_length=80,
                   num_return_sequences=3,
                   temperature=0.8,
                   top_k=50,
                   top_p=0.95)

for i, sample in enumerate(output):
    print(f"\nGenerated text {i+1}:\n{sample['generated_text']}")

# Experiment: change temperature & top_p
print("\n--- Experiment: temperature=1.5, top_p=0.7 ---")
output_exp = generator(prompt,
                       max_length=80,
                       num_return_sequences=2,
                       temperature=1.5,
                       top_p=0.7)
for i, sample in enumerate(output_exp):
    print(f"\nExp {i+1}:\n{sample['generated_text']}")


# ---------- Part 4: Compare Different Models ----------
print("\n=== Part 4: Compare Models ===")
models = ["gpt2", "distilgpt2"]
prompt = "Climate change is one of the most critical issues because"

for m in models:
    print(f"\n--- Results from {m} ---")
    gen = pipeline("text-generation", model=m)
    print(gen(prompt, max_length=60, num_return_sequences=1)[0]["generated_text"])


# ---------- Part 5: Creative Generation Task ----------
print("\n=== Part 5: Creative Generation ===")
creative_prompts = [
    "Success comes when you...",
    "Breaking News: Artificial Intelligence has..."
]

best_idx = None
best_text = ""
for idx, p in enumerate(creative_prompts, 1):
    set_seed(42)
    txt = generator(p, max_length=70, num_return_sequences=1)[0]["generated_text"]
    print(f"\n{idx}. {txt}")
    if best_idx is None:
        best_idx, best_text = idx, txt

print("\n**Best one:** #", best_idx)
print(best_text)
print("Justification: It is concise, realistic, and captures a plausible future headline.")


# ---------- Part 6: Prompts in Creative Task ----------
print("\n=== Part 6: Prompt Sensitivity ===")
poetic_prompts = [
    "Write a poem about artificial intelligence.",
    "As a poet, describe how AI transforms human emotions.",
    "Explain the impact of AI in a poetic way."
]

for p in poetic_prompts:
    set_seed(42)
    print(f"\nPrompt: {p}")
    print(generator(p, max_length=80, num_return_sequences=1)[0]["generated_text"])


# ---------- Part 6 (continued): Summarization ----------
print("\n=== Part 6: Summarization ===")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

text = """Artificial Intelligence is transforming education by enabling personalized learning 
paths, automated grading, and real-time feedback. However, ethical and privacy concerns 
remain significant challenges."""

summary = summarizer(text,
                     max_length=40,
                     min_length=20,
                     do_sample=False)

print("Summary:", summary[0]['summary_text'])

# Try a longer article (example from Wikipedia)
long_article = """
Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural 
intelligence displayed by humans and animals. Leading AI textbooks define the field as the study 
of "intelligent agents": any device that perceives its environment and takes actions that maximize 
its chance of successfully achieving its goals. Colloquially, the term "artificial intelligence" 
is often used to describe machines that mimic "cognitive" functions that humans associate with the 
human mind, such as "learning" and "problem solving".
"""

summary_long = summarizer(long_article,
                          max_length=60,
                          min_length=30,
                          do_sample=False)
print("\nLong-article summary:", summary_long[0]['summary_text'])


# --------------------------------------------------------------
# Creative Mini-project: Short Story Generator
# --------------------------------------------------------------
print("\n=== Creative Mini-project: Short Story Generator ===")

def story_generator(start_prompt: str,
                    model_name: str = "gpt2",
                    max_len: int = 250,
                    temperature: float = 0.9,
                    top_p: float = 0.92,
                    seed: int = 123):
    """
    Generates a short story starting with `start_prompt`.
    """
    set_seed(seed)
    gen = pipeline("text-generation", model=model_name)
    out = gen(start_prompt,
               max_length=max_len,
               temperature=temperature,
               top_p=top_p,
               num_return_sequences=1,
               truncation=True)
    return out[0]["generated_text"]

# ---- Run the mini-project ----
start = "Once upon a time in a futuristic city..."
story = story_generator(start,
                        model_name="gpt2-medium",   # larger model → richer story
                        max_len=300,
                        temperature=0.85,
                        top_p=0.93,
                        seed=42)

print("\n--- Generated Short Story ---")
print(story)

# --------------------------------------------------------------
# End of script
# --------------------------------------------------------------
gen = pipeline("text-generation", model="gpt2-medium", device=0)  # GPU