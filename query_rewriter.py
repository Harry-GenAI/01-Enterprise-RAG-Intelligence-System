from llmservice import call_llm

def rewrite_query(query:str, history:str, mode:str="rewrite")->str:
    
    prompt = f"""
    You are query rewrite assistant.

    Your job is rewrite the user vague query with standalone question 
    which is optimized for semantic search from the knowledge database.

    Rules:
    -Do not change the meaning
    -Do not explain rewrite
    -Do not ask follow-up questions
    -Do not answer the question
    -check for grammatical errors
    -Use chat history below for vague words like "it", "that", "there" or this.
    -If the query is vague use the most relevant topic in the chat history and rewrite the query
    -If user asked a new topic question then dont involve the past chat history topic into it

    Examples:
    
    Chat History: user asked about HR leave policy
    user: what about that?
    rewritten query: What is company's HR leave policy?

    Chat History: user asked about HR leave policy
    user:what is refund time?
    rewritten query: what is the refund processing time?

    Chat History:
    {history}

    User Query:
    {query}

    Rewritten query:
    """

    response = call_llm(prompt=prompt)

    return response.strip()