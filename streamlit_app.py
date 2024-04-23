from rag import initialize_session_storage, read_from_session_state, save_file, save_msg, query_engine_from_doc
import streamlit as st

initialize_session_storage()
container = st.container(height=520, border=True)
with container:
    read_from_session_state()
if file := st.file_uploader("choose a file", type="pdf"):
    if save_file(file=file):
        query_engine = query_engine_from_doc([f"files/{file.name}"])
try:
    if query := st.chat_input("ask document?"):
        with container:
            with st.chat_message("user"):
                st.write(query)
            save_msg("user", query)
            with st.chat_message("assistant"):
                with st.spinner(""):
                    response = query_engine.query(query).response
                    st.write(response)
            save_msg("assistant", response)
except:
    st.error("Please choose your file!", icon="😤")
