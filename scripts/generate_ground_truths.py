"""Regenerate ground-truth answers for the RAGAS eval dataset.

The original eval_questions.json ships with placeholder ground truths like
"Sourced from Apple's 10-K. The exact figure will be populated from the
actual downloaded filing." These make RAGAS Context Recall / Answer
Relevancy unusable.

This script:
  1. Loads each question
  2. Retrieves top-k=20 chunks from ChromaDB (wide net)
  3. Asks GPT-4o-mini to answer the question using ONLY the retrieved text
  4. Saves the answer as the new `ground_truth` field

Cost: ~$0.05 for 20 questions with gpt-4o-mini.

Usage:
    python scripts/generate_ground_truths.py
    python scripts/generate_ground_truths.py --output evaluation/datasets/eval_questions_v2.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_openai import ChatOpenAI

from src.config import settings
from src.retrieval.retriever import RBACRetriever


SYSTEM_PROMPT = """You are generating reference answers for a RAG evaluation dataset.

You will be given a question and excerpts from SEC 10-K filings. Generate a concise,
factual answer (2-4 sentences) based ONLY on the provided excerpts. Include specific
numbers, names, and citations when present. If the excerpts don't contain the answer,
say "The provided filing excerpts do not contain enough information to answer."

Do not hedge, do not add disclaimers, do not mention you are reading excerpts.
Just give the factual answer as a financial analyst would write it.
"""


def generate_ground_truth(question: str, access_roles: list[str], llm: ChatOpenAI) -> str:
    # Retrieve widely — top_k=20 to give GPT plenty of context
    retriever = RBACRetriever(user_roles=set(access_roles), top_k=20)
    docs = retriever.retrieve(question)

    if not docs:
        return "The provided filing excerpts do not contain enough information to answer."

    context = "\n\n---\n\n".join(
        f"[{d.metadata.get('ticker', '?')} {d.metadata.get('section_name', '?')}]\n{d.page_content}"
        for d in docs
    )

    prompt = f"""Question: {question}

Filing Excerpts:
{context}

Answer (2-4 sentences, factual, with specific numbers/names when present):"""

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ])
    return response.content.strip()


def main(input_path: str, output_path: str) -> None:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=settings.openai_api_key,
        temperature=0.0,
    )

    with open(input_path) as f:
        questions = json.load(f)

    print(f"Regenerating ground truths for {len(questions)} questions...")
    print(f"Source: {input_path}")
    print(f"Output: {output_path}")
    print()

    for i, item in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {item['question'][:70]}...")
        try:
            new_gt = generate_ground_truth(
                question=item["question"],
                access_roles=item["access_roles"],
                llm=llm,
            )
            item["ground_truth_original"] = item["ground_truth"]
            item["ground_truth"] = new_gt
            print(f"    -> {new_gt[:120]}{'...' if len(new_gt) > 120 else ''}")
        except Exception as e:
            print(f"    ERROR: {e}")

    with open(output_path, "w") as f:
        json.dump(questions, f, indent=2)

    print(f"\nSaved {len(questions)} questions with regenerated ground truths to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="evaluation/datasets/eval_questions.json")
    parser.add_argument("--output", default="evaluation/datasets/eval_questions_v2.json")
    args = parser.parse_args()
    main(args.input, args.output)
