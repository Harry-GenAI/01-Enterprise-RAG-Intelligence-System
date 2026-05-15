import json
import boto3
from logger import logger

# Bedrock client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1"
)

def call_llm(prompt: str, mode: str = "answer", temperature: float | None = None):
    """
    mode:
    - "rewrite" -> fast + cheap (Haiku)
    - "answer"  -> smart + reasoning (Mistral Large 3)
    """
    
    if mode == "rewrite":
        # Keep Haiku for quick query rephrasing
        model_id = "anthropic.claude-3-haiku-20240307-v1:0"
        logger.info("Calling Haiku (Query Rewriter)")
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 500,
            "temperature": 0.3 if temperature is None else temperature
        }
    else:
        # Upgrading to Mistral Large 3 for the final answer
        model_id = "mistral.mistral-large-2402-v1:0" 
        logger.info("Calling Mistral Large 3 (Answer Generation)")
        
        # Mistral uses a different prompt format (Instruction tags)
        body = {
            "prompt": f"<s>[INST] {prompt} [/INST]",
            "max_tokens": 1000,
            "temperature": 0.5 if temperature is None else temperature
        }

    response = bedrock.invoke_model(
        modelId=model_id,
        body=json.dumps(body),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read())
    
    # Logic to handle different response structures
    if "mistral" in model_id:
        # Mistral returns text in 'outputs'
        return result["outputs"][0]["text"].strip()
    else:
        # Claude returns text in 'content'
        return result["content"][0]["text"].strip()

