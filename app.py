import streamlit as st
import config
from document_loader import SUPPORTED_TYPES, save_uploaded_file, load_document
from ingest import add_documents, load_doc_store, clear_all, get_file_summaries, remove_file
from chatbot import answer_question

st.set_page_config(page_title="Advance RAG")
st.title("Chat with your documents")

if "messages" not in st.session_state:
    st.session_state.messages = []


# Sidebar: upload and manage documents
with st.sidebar:
    st.header("Documents")

    if not config.LLM_API_KEY:
        st.error("LLM_API_KEY is missing. Add it to the .env file and restart the app.")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX or TXT files",
        type=SUPPORTED_TYPES,
        accept_multiple_files=True
    )

    if st.button("Process", disabled=not uploaded_files):
        added_files = load_doc_store()["files"]
        new_files = [f for f in uploaded_files if f.name not in added_files]

        if not new_files:
            st.info("These files are already added.")
        else:
            with st.spinner("Extracting text, chunking and creating embeddings..."):
                documents = []
                file_names = []

                for uploaded_file in new_files:
                    try:
                        file_path = save_uploaded_file(uploaded_file)
                        documents += load_document(file_path)
                        file_names.append(uploaded_file.name)
                    except Exception as e:
                        st.warning(f"Could not read {uploaded_file.name}: {e}")

                if documents:
                    num_parents, num_children = add_documents(documents, file_names)
                    st.success(f"Added {len(file_names)} file(s): {num_parents} parent and {num_children} child chunks.")
                else:
                    st.error("No text was found in these files.")

            if documents:
                with st.spinner("Writing a short overview of each file..."):
                    get_file_summaries()

    added_files = load_doc_store()["files"]
    if added_files:
        st.subheader("Added files")
        summaries = get_file_summaries()

        for name in added_files:
            with st.expander(name):
                st.write(summaries.get(name, "No overview yet."))
                if st.button("Remove this file", key=f"remove_{name}"):
                    remove_file(name)
                    st.rerun()

        if st.button("Remove all documents"):
            clear_all()
            st.session_state.messages = []
            st.rerun()

    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

    with st.expander("Settings"):
        st.json(config.describe_settings())


# Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("Sources: " + ", ".join(message["sources"]))

question = st.chat_input("Ask a question about your documents")

if question:
    if not load_doc_store()["files"]:
        st.warning("Please upload and process at least one document first.")
    else:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    stream, get_sources = answer_question(question, st.session_state.messages)
                answer = st.write_stream(stream)
                sources = get_sources(answer)
                if sources:
                    st.caption("Sources: " + ", ".join(sources))
            except Exception as e:
                # for example a wrong API key, a rate limit or no internet; the failed turn is not saved
                st.error(f"Sorry, something went wrong while answering. Please try again.\n\nDetails: {e}")
                answer = None

        if answer:
            st.session_state.messages.append({"role": "user", "content": question})
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
