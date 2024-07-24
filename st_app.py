import os
from glob import glob
import streamlit as st
from dotenv import load_dotenv
from utils import get_langchain_model, save_uploadedfile


load_dotenv(".env")


def main():
    st.set_page_config(page_title="Apple Vision Pro Sales Agent")
    FILES_PATH = "temp_doc_store"

    ### Side Bar
    with st.sidebar:
        st.title("Apple Vision Pro Sales Agent")

        files = st.file_uploader(
            "Select files to Chat with!",
            accept_multiple_files=True,
        )

        file_names = [_file.name for _file in files]
        btn = st.button("Submit Files")

        st.write("---")

        url_in = st.text_input("Enter URL's with comma seperation")
        urls = [_url for _url in url_in.split(",")]
        btn_url = st.button("Submit URL's")

        st.write("---")

        yt_url_in = st.text_input("Enter Youtube URL's with comma seperation")
        yt_urls = [_url for _url in yt_url_in.split(",")]
        btn_yt_url = st.button("Submit Youtube URL's")

        st.write("---")

        clear_chat = st.button("Clear Chat", type="primary")

    chat = st.chat_input("Enter the question")

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "How may I help you?"}
        ]

    if "history" not in st.session_state.keys():
        st.session_state.history = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if btn and len(files) > 0:
        for _file in files:
            save_uploadedfile(_file, FILES_PATH)
            file_paths = glob(os.path.join(FILES_PATH, "**"))
        with st.chat_message("assistant"):
            with st.spinner("Processing Files....."):
                st.session_state.agent = get_langchain_model(docs=file_paths)

                response = f"Processed the following files: {file_names}\n\nPlease ask the questions!"
                st.write(response)
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

    if btn_url and len(urls) > 0:
        with st.chat_message("assistant"):
            with st.spinner("Processing URL's....."):
                st.session_state.agent = get_langchain_model(urls=urls)
                response = (
                    f"Processed the following urls: {urls}\n\nPlease ask the questions!"
                )
                st.write(response)

    if btn_yt_url and len(yt_urls) > 0:
        with st.chat_message("assistant"):
            with st.spinner("Processing Youtube URL's....."):
                st.session_state.agent = get_langchain_model(yt_urls=yt_urls)
                response = f"Processed the following urls: {yt_urls}\n\nPlease ask the questions!"
                st.write(response)

    ### Chat Interface

    if chat:
        if st.session_state.get("agent", None) is None:
            file_paths = glob(os.path.join(FILES_PATH, "**"))
            st.session_state.agent = get_langchain_model(file_paths)
        st.session_state.messages.append({"role": "user", "content": chat})
        with st.chat_message("user"):
            st.write(chat)
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    final_response = {"output": None}

                    res = st.session_state.agent(chat)
                    st.session_state.history.append(chat)
                    final_response["output"] = res["output"]

                    st.write(final_response["output"])

            message = {"role": "assistant", "content": final_response["output"]}
            st.session_state.messages.append(message)

    if clear_chat:
        st.session_state.messages.clear()
        st.session_state.messages = [
            {"role": "assistant", "content": "How may I help you?"}
        ]


if __name__ == "__main__":
    main()
