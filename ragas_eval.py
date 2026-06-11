import asyncio

from dotenv import load_dotenv


load_dotenv()

from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate

from rag import retrieve_context
from prompts import build_prompt
from llmservice import call_llm

# Inputs
questions = [
    "What is refund processing time?",
    "Can employees misuse company systems?",
    "Are unauthorized downloads allowed for employees?",
    "What is the policy for confidential data sharing?",
    "How often should passwords be rotated?"
]

ground_truths = [
    "Refund processing time is 5-7 business days.",
    "Employees must use company systems responsibly.",
    "Unauthorized downloads are strictly prohibited.",
    "The policy is that confidential data must not be shared externally.",
    "Passwords must be rotated every 90 days."
]

# Ragas Pipeline
async def run_ragas():
    answers = []
    contexts = []

    for idx, query in enumerate(questions, start=1):
        # get contexts
        context, _, _debug_results = await asyncio.to_thread(retrieve_context, query, None)

        # get prompt
        prompt = build_prompt("", context, query)

        # get answer
        answer = await asyncio.to_thread(call_llm, prompt, "answer", temperature=0)

        # collect generated answers and contexts
        answers.append(answer)
        contexts.append([context])

        print(f"\n--- Sample {idx} ---")
        print(f"Question: {query}")
        print(f"Ground truth: {ground_truths[idx-1]}")
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
    results = evaluate(
        dataset,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
    )

    print("\nRAGAS Results:\n")
    print(results)

    print("\nRAGAS Row-level Results:\n")
    row_results = results.to_pandas()
    metric_columns = [
        "answer_relevancy",
        "context_precision",
        "faithfulness",
        "context_recall",
    ]
    metric_results = row_results[metric_columns].round(3)
    metric_results.index = metric_results.index + 1 #to make dataframe index starts from 1
    print(metric_results.to_string(index=True))


# main
if __name__ == "__main__":
    asyncio.run(run_ragas())
