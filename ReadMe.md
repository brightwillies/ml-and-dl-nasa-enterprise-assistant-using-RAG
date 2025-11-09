# NASA Policy Query Assistant  
**An AI-powered RAG Chatbot for NASA Policy Documents**  
*Group 5 – Enterprise Knowledge Management Project*  
**Members:** Aryan Ajgaonkar, Bernice Eghan, Bright Williams Boakye, Jeffrey Okoduwa, Joy Musa, Simranpreet Kaur  

---

## Project Overview  
This project implements a **Retrieval-Augmented Generation (RAG)** system to enable **fast, accurate, and cited answers** to queries about NASA policies, SOPs, and procedural requirements. Built using **LangChain, Streamlit, OpenAI, and ChromaDB**, the assistant retrieves relevant document chunks from a vector database and generates responses grounded in official NASA sources.

<!-- > **Live Demo (if deployed):** [https://your-streamlit-app-url](https://your-streamlit-app-url) *(replace with actual link)*   -->

---

## Key Features  
- **Semantic Search** over 14,552 embedded chunks from 212 NASA policy PDFs  
- **Conversational UI** with chat history and follow-up support  
- **Page-specific citations** with clickable links to NODIS PDFs  
- **Query routing** by topic (e.g., Budgeting, Audits, Partnerships)  
- **Recency-aware ranking** for time-sensitive policies  
- **Performance logging** (`rag_metrics.log`, `rag_errors.log`)  
- **Local vector database** (ChromaDB) – no external API needed for retrieval  

---

## System Architecture (Step-by-Step Pipeline)

1. **User Query Input**  
   - User enters a natural language query via Streamlit's `st.chat_input`  
   - Example: `"NASA's budgeting procedures"`

2. **Query Routing**  
   - `route_query()` uses regex to classify the topic  
   - Applies **filter** (e.g., `series=Financial Management`) and **ranking** (e.g., recency)

3. **Semantic Retrieval**  
   - Query is embedded using `text-embedding-ada-002`  
   - ChromaDB performs **vector search** with `k=15` and metadata filtering  
   - Returns top 15 relevant document chunks with `category`, `page_label`, and `timestamp`

4. **Prompt Construction**  
   - Retrieved chunks are formatted with **page-specific URLs**  
   - Chat history (last 10 messages) is added for context-aware follow-ups  
   - Final prompt: `{query} + {context} + {history}`

5. **Answer Generation**  
   - `gpt-4o-mini` (temperature=0) generates a **concise, cited response**  
   - Output includes inline references (e.g., "N_PR_9420_001A, Page 4")

6. **UI Display & Citations**  
   - Answer shown in `st.chat_message`  
   - **Cited Documents** section with clickable NODIS links  
   - Chat history updated in `st.session_state`

7. **Performance Logging**  
   - Metrics logged to `rag_metrics.log`:  
     - Precision, Relevance (1–5), Latency  
   - Errors logged to `rag_errors.log`

8. **Follow-Up Ready**  
   - User can ask follow-up questions  
   - Full context retained for coherent conversation