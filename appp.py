import streamlit as st
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import logging
import re
from datetime import datetime
import time
from functools import lru_cache

# Set up logging for errors and metrics
logging.basicConfig(filename="rag_errors.log", level=logging.ERROR,
                    format="%(asctime)s - %(levelname)s - %(message)s")
metrics_logger = logging.getLogger("rag_metrics")
metrics_handler = logging.FileHandler("rag_metrics.log")
metrics_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
metrics_logger.addHandler(metrics_handler)
metrics_logger.setLevel(logging.INFO)

# Initialize embeddings and LLM
# embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
# llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

try:
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-ada-002",
        openai_api_key=st.secrets["OPENAI_API_KEY"]
    )
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=st.secrets["OPENAI_API_KEY"]
    )
except KeyError:
    st.error("OPENAI_API_KEY not found. Please add it in Streamlit Secrets.")
    st.stop()
    
# Load vector store
PERSISTENT_DIR = "./chroma_db_full"
try:
    vectorstore = Chroma(
        collection_name="nasa_policies",
        embedding_function=embeddings_model,
        persist_directory=PERSISTENT_DIR
    )
    st.write(f"Loaded vector store with {vectorstore._collection.count()} documents")
except Exception as e:
    logging.error(f"Vector store error: {e}")
    st.error(f"Failed to load vector store: {e}")
    st.stop()

# Prompt template
prompt_template = PromptTemplate.from_template(
    """You are a knowledgeable assistant for NASA policy queries. Use the following context to answer the query concisely and accurately. Cite the source PDF and page number for each piece of information used. If the context is insufficient, say so and provide a general answer based on available information.

    Context:
    {context}

    Query: {query}

    Answer:
    """
)


# Format context with hyperlinks
def format_context(docs):
    context = []
    for doc in docs:
        category = doc.metadata['category']
        # Remove any trailing suffixes like __main for citation
        clean_category = re.sub(r'__main$', '', category)
        page_label = doc.metadata['page_label']
        score = doc.metadata.get('score', 'N/A')
        # Generate hyperlink
        url = f"https://nodis3.gsfc.nasa.gov/npg_img/{clean_category}/{clean_category}.pdf"
        citation = f"[{clean_category} (Page {page_label}, Similarity Score: {score})]({url})"
        context.append(f"Source: {citation}:\n{doc.page_content}")
    return "\n\n".join(context)


# Ranking function
def rank_documents(docs, sort_by="score", reverse=True):
    for i, doc in enumerate(docs):
        doc.metadata['score'] = 1.0 - (i * 0.05)  # Adjusted score decrement for k=15
    if sort_by == "recency":
        docs.sort(key=lambda doc: doc.metadata['creation_timestamp'], reverse=True)
    return docs


# Create RAG chain with caching
@lru_cache(maxsize=100)
def create_rag_chain_cached(query, filter_dict, ranking):
    # Convert filter_dict back to dict if not None, else use None
    filter_dict = dict(filter_dict) if filter_dict else None
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 15, "filter": filter_dict}  # Increased k to 15
    )
    chain = (
            {"context": retriever | rank_documents | format_context, "query": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
    )
    return chain.invoke(query)


# Query router
def route_query(query):
    if re.search(r'\b(audit|audits|contract audit|contract audits|fraud|inspector general)\b', query, re.IGNORECASE):
        return {
            "topic": "Audits and Investigations",
            "filter": {"series": "Audits and Investigations"},
            "ranking": "score",
            "description": "Audits and Investigations, rank by similarity"
        }
    elif re.search(r'\b(financial reporting|report|accounting)\b', query, re.IGNORECASE):
        return {
            "topic": "Financial Reporting",
            "filter": {"category": "N_PD_9501_001I"},
            "ranking": "score",
            "description": "Financial Reporting (N_PD_9501_001I), rank by similarity"
        }
    elif re.search(r'\b(budget|budgeting|funds|appropriation)\b', query, re.IGNORECASE):
        return {
            "topic": "Budgeting",
            "filter": {"series": "Financial Management"},
            "ranking": "recency",
            "description": "Budgeting (Financial Management), rank by recency"
        }
    elif re.search(r'\b(partnership|partnerships|agreement|agreements|collaboration)\b', query, re.IGNORECASE):
        return {
            "topic": "Partnership Agreements",
            "filter": {"category": "N_PD_1050_007A"},
            "ranking": "score",
            "description": "Partnership Agreements (N_PD_1050_007A), rank by similarity"
        }
    else:
        return {
            "topic": "General",
            "filter": None,
            "ranking": "score",
            "description": "General query, no filter, rank by similarity"
        }


# Streamlit UI
st.title("NASA Policy Query Assistant")
st.write("Enter a query about NASA policies, and the assistant will provide a concise answer with clickable citations.")

query = st.text_input("Enter your query:", "")
if query:
    start_time = time.time()
    try:
        route = route_query(query)
        st.write(f"Routed to: {route['description']}")
        # Pass filter as dict or None to cache
        filter_dict = route["filter"] if route["filter"] else ""
        answer = create_rag_chain_cached(query, tuple(filter_dict.items()) if filter_dict else (), route["ranking"])

        # Render answer with markdown for hyperlinks
        st.markdown(answer, unsafe_allow_html=True)

        # Log metrics
        retriever = vectorstore.as_retriever(search_kwargs={"k": 15, "filter": route["filter"]})
        docs = retriever.invoke(query)
        precision = 1.0 if docs else 0.0  # Simplified precision for UI
        relevance_score = 5 if "N/A" not in answer and "insufficient" not in answer.lower() else 3
        latency = time.time() - start_time
        metrics_logger.info(
            f"Query: '{query}' | Precision: {precision:.2f} | Relevance: {relevance_score}/5 | Latency: {latency:.2f}s"
        )
    except Exception as e:
        logging.error(f"RAG error for query '{query}': {e}")
        st.error(f"Failed to process query: {e}")