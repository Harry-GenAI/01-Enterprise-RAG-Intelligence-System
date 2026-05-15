import asyncio
import re

from dotenv import load_dotenv


load_dotenv()

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.run_config import RunConfig

from rag import retrieve_context
from prompts import build_prompt
from llmservice import call_llm

# Inputs
questions = [
    "What is refund processing time?",
    "Can employees misuse company systems?",
    "Are unauthorized downloads allowed for employees?",
    "Can confidential data be shared?",
    "How often should passwords be rotated?"
]

ground_truths = [
    "Refund processing time is 5-7 business days.",
    "Employees must use company systems responsibly.",
    "Unauthorized downloads are strictly prohibited.",
    "Confidential data must not be shared externally.",
    "Passwords must be rotated every 90 days."
]


def split_context_blocks(context: str) -> list[str]:
    """Keep RAGAS contexts as separate source blocks instead of one huge string."""
    blocks = [
        block.strip()
        for block in re.split(r"(?=\[SOURCE\s*:)", context)
        if block.strip()
    ]
    return blocks or [context]


# Ragas Pipeline
async def run_ragas():
    answers = []
    contexts = []

    for idx, query in enumerate(questions):
        # get contexts
        context, _, _debug_results = await asyncio.to_thread(retrieve_context, query, None)

        # get prompt
        prompt = build_prompt("", context, query)

        # get answer
        answer = await asyncio.to_thread(call_llm, prompt, "answer", temperature=0)

        # collect generated answers and contexts
        answers.append(answer)
        contexts.append(split_context_blocks(context))

        print(f"\n--- Sample {idx} ---")
        print(f"Question: {query}")
        print(f"Ground truth: {ground_truths[idx]}")
        print(f"Answer: {answer}")
        print(f"Context:\n{context}")
        
    # Dataset
    dataset = Dataset.from_dict({
        "question":questions,
        "ground_truth":ground_truths,
        "answer":answers,
        "contexts":contexts

    })

    # Evaluate
    evaluator_llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        timeout=60,
        max_retries=5,
    )
    evaluator_embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        timeout=60,
        max_retries=5,
    )
    run_config = RunConfig(timeout=120, max_retries=3, max_workers=2)
    results = evaluate(
        dataset,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=run_config,
        raise_exceptions=False,
    )

    print("\nRAGAS Results:\n")
    print(results)

    print("\nRAGAS Row-level Results:\n")
    row_results = results.to_pandas()
    metric_columns = [
        "question",
        "answer_relevancy",
        "context_precision",
        "faithfulness",
        "context_recall",
    ]
    print(row_results[metric_columns].to_string(index=True))


# main
if __name__ == "__main__":
    asyncio.run(run_ragas())
