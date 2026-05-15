PROMPT_VERSION = "enterprise_v2_inference"

def build_prompt(history, context, question):
    """
    Main QA prompt. Updated to allow for policy inference while 
    staying grounded in the Knowledge Context.
    """
    return f"""
    PROMPT_VERSION: {PROMPT_VERSION}

    You are an expert Enterprise Assistant. Your goal is to answer the User Question 
    using the Knowledge Context provided below. 

    INSTRUCTIONS:
    1. Use the Knowledge Context as your PRIMARY source.
    2. If the exact answer isn't in the text, use logical inference based on the policies. 
       (e.g., If context says 'use systems responsibly', you can infer that 'using them in your own way' for non-business tasks is not allowed).
    3. Use the Conversation History to maintain context but prioritize the Knowledge Context for facts.
    4. If the context absolutely does not cover the topic at all, state that "you don't have enough information".
    5. Keep your answer professional, concise, and direct.
    6. Start with the direct answer and mirror the key terms from the User Question.
       Avoid preambles like "Based on the context" or "According to the provided policy" unless necessary.

    Conversation History:
    {history}

    Knowledge Context:
    {context}

    User Question: 
    {question}

    Answer:"""
