import json
import random
import os

os.makedirs("data/processed", exist_ok=True)

inputs = [
    "i feel depressed",
    "i feel anxious all the time",
    "i cant cope anymore",
    "i feel overwhelmed",
    "i feel empty",
]

crisis_inputs = [
    "i want to kill myself",
    "i want to hurt myself",
    "i dont want to live anymore",
]

def normal_response():
    return (
        "I'm sorry you're feeling this way. It would really help to speak with a mental health professional "
        "or someone you trust."
    )

def crisis_response():
    return (
        "I'm really sorry you're feeling this way. You’re not alone, but this is serious. "
        "Please seek immediate help from a professional or contact your local emergency number."
    )

samples = []

for i in range(10000):
    if random.random() < 0.3:
        user = random.choice(crisis_inputs)
        resp = crisis_response()
    else:
        user = random.choice(inputs)
        resp = normal_response()

    samples.append({
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": resp}
        ],
        "meta": {"type": "mental_health"}
    })

with open("data/processed/mental_health.jsonl", "w") as f:
    for s in samples:
        f.write(json.dumps(s) + "\n")

# Run
# python scripts/edge_cases/mental_health.py