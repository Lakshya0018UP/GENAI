# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_classic.chains import RetrievalQA

# from retriver import get_retriver


# load_dotenv()

# def run_assistant():
#     retriever=get_retriver()

#     llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    

    
#     qa_chain=RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True
#     )


#     query = input("\nAsk your Knowledge Assistant: ")
#     response = qa_chain.invoke({"query": query})
    
#     print(f"\nAnswer: {response['result']}")
#     print("\nSources Used:")
#     for doc in response["source_documents"]:
#         print(f"- {doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')})")

# if __name__ == "__main__":
#     run_assistant()


import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate  # New import for prompt engineering
from retriver import get_retriver

load_dotenv()

def run_assistant():
    # Requirement: Implement complete RAG flow
    retriever = get_retriver()

    # Initializing Gemini with temperature 0 for factual consistency
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

    # Requirement: Implement prompt engineering with clear system instructions
    # Requirement: "Answer only from context" enforcement
    template = """You are a specialized Knowledge Assistant for this GenAI assignment. 
    Use the provided context to answer the user's question.

    Guidelines for your answer:
    - Provide a detailed and descriptive explanation based on the context.
    - If the context allows, break your answer into logical sections or bullet points.
    - Use technical terms from the context where appropriate.
    - Strictly answer ONLY from the context. If the information is not present, say you don't know.
    
    Context: {context}
    Question: {question}
    
    Detailed Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    # Requirement: Response generation with source document references
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        # Injecting the guardrail prompt as required
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} 
    )

    # Interaction method: Command-line interface (CLI)
    while True:
        query = input("\nAsk your Knowledge Assistant: ")
        response = qa_chain.invoke({"query": query})
        if query=="exit":
            break
        print(f"\nAnswer: {response['result']}")
        print("\nSources Used:")
        for doc in response["source_documents"]:
        # metadata extraction from vector search results
            print(f"- {doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')})")

    
    # Requirement: Responses must include source document references
    
if __name__ == "__main__":
    run_assistant()