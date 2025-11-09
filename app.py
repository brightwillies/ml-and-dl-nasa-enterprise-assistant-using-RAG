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

# Updated Prompt template with history
prompt_template = PromptTemplate.from_template(
    """You are a knowledgeable assistant for NASA policy queries.

    Previous conversation:
    {history}

    Context:
    {context}

    Query: {query}

    Instructions:
    1. First, determine if the query is related to NASA policies.
    2. If NOT related to NASA policies, respond with: "This question is outside the scope of NASA policies. I can only help with NASA policy-related queries."
       DO NOT cite any sources in this case.

    3. If related to NASA policies, check if the context contains relevant information.
    4. If the context DOES NOT contain the information needed, respond with: "I don't have information about this in the available NASA policy documents."
       DO NOT cite any sources in this case.

    5. ONLY if the context contains relevant information, provide the answer and cite sources in the format: [Source: filename.pdf, Page: X]

    Remember: Citations should ONLY appear when you are providing information directly from the context. No citations for out-of-scope queries or when information is unavailable.

    Answer:
    """
)


# Generate correct NASA PDF URL with page anchor
def generate_nasa_pdf_url(category, page_label):
    base = "https://nodis3.gsfc.nasa.gov/npg_img/"
    if category.endswith("__main"):
        directory = category.replace("__main", "") + "_/"
        file_name = category + ".pdf#page=" + str(page_label)
    elif category.endswith("_"):
        directory = category + "/"
        file_name = category + ".pdf#page=" + str(page_label)
    else:
        directory = category + "/"
        file_name = category + ".pdf#page=" + str(page_label)
    return base + directory + file_name


# Format context with hyperlinks and collect cited documents
def format_context(docs):
    context = []
    cited_documents = set()  # Use set to avoid duplicates
    for doc in docs:
        category = doc.metadata['category']
        # Remove any trailing suffixes like __main for citation display
        clean_category = re.sub(r'__main$', '', category)
        page_label = doc.metadata['page_label']
        score = doc.metadata.get('score', 'N/A')
        # Generate hyperlink using the new function
        url = generate_nasa_pdf_url(category, page_label)
        citation = f"[{clean_category} (Page {page_label}, Similarity Score: {score})]({url})"
        context.append(f"Source: {citation}:\n{doc.page_content}")
        # Collect cited document
        cited_documents.add((clean_category, url))
    return "\n\n".join(context), list(cited_documents)


# Format chat history for prompt
def format_history(messages):
    history = []
    for msg in messages[-10:]:  # Limit to last 10 messages
        if msg["role"] == "user":
            history.append(f"User: {msg['content']}")
        elif msg["role"] == "assistant":
            history.append(f"Assistant: {msg['content']}")
    return "\n".join(history)


# Ranking function
def rank_documents(docs, sort_by="score", reverse=True):
    for i, doc in enumerate(docs):
        doc.metadata['score'] = 1.0 - (i * 0.05)  # Adjusted score decrement for k=15
    if sort_by == "recency":
        docs.sort(key=lambda doc: doc.metadata['creation_timestamp'], reverse=True)
    return docs


# Create RAG chain with caching
@lru_cache(maxsize=100)
def create_rag_chain_cached(query, filter_dict, ranking, history):
    # Convert filter_dict back to dict if not None, else use None
    filter_dict = dict(filter_dict) if filter_dict else None
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 15, "filter": filter_dict}  # Increased k to 15
    )
    chain = (
            {
                "context": lambda x: format_context(retriever.invoke(x))[0],
                "query": RunnablePassthrough(),
                "history": lambda x: history
            }
            | prompt_template
            | llm
            | StrOutputParser()
    )
    return chain.invoke(query), retriever.invoke(query)


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
st.title("NASA Policy Navigator")
st.write("Enter a query about NASA policies, and the assistant will provide a concise answer with clickable citations.")

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for selected question
if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""

# Initialize session state for auto-process flag
if "process_question" not in st.session_state:
    st.session_state.process_question = False

# === SIDEBAR: Options + Example Questions ===
with st.sidebar:
    st.header("Options")

    # CLEAR CHAT BUTTON (FIXED)
    if st.button("Clear Chat History", disabled=len(st.session_state.messages) == 0):
        st.session_state.messages = []
        st.session_state.selected_question = ""
        st.session_state.process_question = False
        st.success("Chat history cleared!")
        st.rerun()  # Refresh UI immediately

    # Display chat stats
    if st.session_state.messages:
        st.write(f"Conversation length: {len([m for m in st.session_state.messages if m['role'] == 'user'])} messages")

    # === EXAMPLE QUESTIONS PANEL ===
    st.markdown("---")
    st.subheader("Try These Questions")
    example_queries = [
        "What are NASA's budgeting procedures?",
        "How are contract audits conducted?",
        "What is the policy on partnership agreements?",
        "Explain financial reporting requirements",
        "What are the rules for acceptable use of IT equipment?"
    ]

    for q in example_queries:
        if st.button(q, key=q, use_container_width=True):
            st.session_state.selected_question = q
            st.session_state.process_question = True
            st.rerun()

# Display chat messages in a conversational format
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])
            if message.get("cited_documents"):
                st.markdown("**Cited Documents:**")
                for category, url in message["cited_documents"]:
                    st.markdown(f"• [{category}]({url})")

# Process example questions (only if not already processed)
if st.session_state.process_question and st.session_state.selected_question:
    # Check if this question was already answered in the last message
    last_user_msg = None
    last_assistant_msg = None

    # Find the last user and assistant messages
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "user" and last_user_msg is None:
            last_user_msg = msg
        elif msg["role"] == "assistant" and last_assistant_msg is None:
            last_assistant_msg = msg

    # Only process if the last user message is different from the selected question
    # or if there's no assistant response yet
    should_process = True
    if last_user_msg and last_user_msg["content"] == st.session_state.selected_question:
        if last_assistant_msg:
            # Already answered, don't process again
            should_process = False

    if should_process:
        # Add the selected question to messages
        st.session_state.messages.append({"role": "user", "content": st.session_state.selected_question})

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(st.session_state.selected_question)

        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")

            start_time = time.time()
            try:
                # Route query
                route = route_query(st.session_state.selected_question)

                # Pass filter and history
                filter_dict = route["filter"] if route["filter"] else ""
                history = format_history(st.session_state.messages)

                # Get answer and documents
                answer, docs = create_rag_chain_cached(
                    st.session_state.selected_question,
                    tuple(filter_dict.items()) if filter_dict else (),
                    route["ranking"],
                    history
                )

                # Get cited documents
                _, cited_documents = format_context(docs)

                # Display final answer
                message_placeholder.markdown(answer)

                # Display cited documents
                if cited_documents:
                    st.markdown("**Cited Documents:**")
                    for category, url in cited_documents:
                        st.markdown(f"• [{category}]({url})")

                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "cited_documents": cited_documents
                })

                # Log metrics
                precision = 1.0 if docs else 0.0
                relevance_score = 5 if "N/A" not in answer and "insufficient" not in answer.lower() else 3
                latency = time.time() - start_time
                metrics_logger.info(
                    f"Query: '{st.session_state.selected_question}' | Precision: {precision:.2f} | Relevance: {relevance_score}/5 | Latency: {latency:.2f}s"
                )

            except Exception as e:
                error_msg = f"Sorry, I encountered an error processing your query: {str(e)}"
                message_placeholder.markdown(error_msg)
                logging.error(f"RAG error for query '{st.session_state.selected_question}': {e}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "cited_documents": []
                })

    # Reset the flags
    st.session_state.selected_question = ""
    st.session_state.process_question = False
    st.rerun()

# Chat input at the bottom
if prompt := st.chat_input("Enter your question about NASA policies..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        start_time = time.time()
        try:
            # Route query
            route = route_query(prompt)

            # Pass filter and history
            filter_dict = route["filter"] if route["filter"] else ""
            history = format_history(st.session_state.messages)

            # Get answer and documents
            answer, docs = create_rag_chain_cached(
                prompt,
                tuple(filter_dict.items()) if filter_dict else (),
                route["ranking"],
                history
            )

            # Get cited documents
            _, cited_documents = format_context(docs)

            # Display final answer
            message_placeholder.markdown(answer)

            # Display cited documents
            if cited_documents:
                st.markdown("**Cited Documents:**")
                for category, url in cited_documents:
                    st.markdown(f"• [{category}]({url})")

            # Add assistant response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "cited_documents": cited_documents
            })

            # Log metrics
            precision = 1.0 if docs else 0.0
            relevance_score = 5 if "N/A" not in answer and "insufficient" not in answer.lower() else 3
            latency = time.time() - start_time
            metrics_logger.info(
                f"Query: '{prompt}' | Precision: {precision:.2f} | Relevance: {relevance_score}/5 | Latency: {latency:.2f}s"
            )

        except Exception as e:
            error_msg = f"Sorry, I encountered an error processing your query: {str(e)}"
            message_placeholder.markdown(error_msg)
            logging.error(f"RAG error for query '{prompt}': {e}")
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "cited_documents": []
            })

# Custom CSS for better chat styling
st.markdown(
    """
    <style>
    .stChatMessage {
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)