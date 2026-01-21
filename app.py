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


# import os
# from dotenv import load_dotenv
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_classic.chains import RetrievalQA
# from langchain_classic.prompts import PromptTemplate  # New import for prompt engineering
# from retriver import get_retriver

# load_dotenv()

# def run_assistant():
#     # Requirement: Implement complete RAG flow
#     retriever = get_retriver()

#     # Initializing Gemini with temperature 0 for factual consistency
#     llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

#     # Requirement: Implement prompt engineering with clear system instructions
#     # Requirement: "Answer only from context" enforcement
#     template = """You are a specialized Knowledge Assistant for this GenAI assignment. 
#     Use the provided context to answer the user's question.

#     Guidelines for your answer:
#     - Provide a detailed and descriptive explanation based on the context.
#     - If the context allows, break your answer into logical sections or bullet points.
#     - Use technical terms from the context where appropriate.
#     - Strictly answer ONLY from the context. If the information is not present, say you don't know.
    
#     Context: {context}
#     Question: {question}
    
#     Detailed Answer:"""
    
#     QA_CHAIN_PROMPT = PromptTemplate(
#         input_variables=["context", "question"],
#         template=template,
#     )

#     # Requirement: Response generation with source document references
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=llm,
#         chain_type="stuff",
#         retriever=retriever,
#         return_source_documents=True,
#         # Injecting the guardrail prompt as required
#         chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} 
#     )

#     # Interaction method: Command-line interface (CLI)
#     while True:
#         query = input("\nAsk your Knowledge Assistant: ")
#         response = qa_chain.invoke({"query": query})
#         if query=="exit":
#             break
#         print(f"\nAnswer: {response['result']}")
#         print("\nSources Used:")
#         for doc in response["source_documents"]:
#         # metadata extraction from vector search results
#             print(f"- {doc.metadata['source']} (Page {doc.metadata.get('page', 'N/A')})")

    
#     # Requirement: Responses must include source document references
    
# if __name__ == "__main__":
#     run_assistant()


import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from retriver import get_retriver
from dotenv import load_dotenv

# 1. Setup Page Config & Load Environment
st.set_page_config(page_title="GenAI Knowledge Assistant", layout="centered")
load_dotenv()

# --- CSS to Hide Icons (The "No Emojis" Request) ---
st.markdown("""
    <style>
    [data-testid="stChatMessageAvatarUser"],
    [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="stChatMessageAvatarCustom"] {
        display: none !important;
    }
    [data-testid="stChatMessage"] {
        padding-left: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Cache the RAG Chain
@st.cache_resource
def initialize_rag_system():
    retriever = get_retriver()
    # Ensure you use the model name exactly as per your API access
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    template = """You are a specialized Knowledge Assistant for this GenAI assignment. 
    Use the provided context to answer the user's question accurately.
    

    

    Guidelines for your answer:
    - Provide a detailed and descriptive explanation based on the context.
    - If the context allows, break your answer into logical sections or bullet points.
    - Use technical terms from the context where appropriate.
    - Strictly answer ONLY from the context. If the information is not present, say you don't know.
    
    Context: {context}
    Question: {question}
    
    Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT} 
    )
    return qa_chain

# Initialize the system
qa_chain = initialize_rag_system()

# 3. Streamlit UI Elements
st.title("GenAI Knowledge Assistant")
st.markdown("---") # Visual separator

# 4. Chat Interface Logic
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input & Response Generation
if query := st.chat_input("What would you like to know?"):
    
    # A. Display user message immediately
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # B. Generate and Display Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Processing ..."):
            # The heart of the retrieval process
            response = qa_chain.invoke({"query": query})
            answer = response['result']
            
            st.markdown(answer)
            
            # Display Sources
            with st.expander("View Source References"):
                for doc in response["source_documents"]:
                    source = doc.metadata.get('source', 'Unknown File')
                    page = doc.metadata.get('page', 'N/A')
                    st.write(f"- **{source}** (Page {page})")

    # C. Save assistant message to history for the next rerun
    st.session_state.messages.append({"role": "assistant", "content": answer})