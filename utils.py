import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.agents.types import AgentType
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.document_loaders.youtube import YoutubeLoader

from langchain_community.document_loaders.chromium import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer


def get_langchain_model(docs=None, urls=None, yt_urls=None):
    system_message = """
    Act like a Sales Agent for the Apple Vision Pro glasses and answer everything in a pointer format
    This is conversation with a customer i.e. a Human. Answer the questions you get based on the knowledge you have and what you get from the documents."
    If you don't know the answer, just say that you don't know the answer, don't try to make up an answer."
    Use the internet when required
    """

    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.3,
    )

    bs_transformer = BeautifulSoupTransformer()

    embeddings = OpenAIEmbeddings()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )

    documents = []

    def parse_single_document(f):
        print(f"processing {f}")
        loader = UnstructuredFileLoader(f)

        docs_ = loader.load_and_split(text_splitter)
        documents.extend(docs_)

    def parse_single_yt_video(url):
        print(f"processing {url}")
        loader = YoutubeLoader.from_youtube_url(url)
        docs_ = loader.load_and_split(text_splitter)
        documents.extend(docs_)

    if docs is not None:
        for _d in docs:
            parse_single_document(_d)

    elif yt_urls is not None:
        for _u in yt_urls:
            parse_single_yt_video(_u)

    vectorstore = Chroma.from_documents(documents, embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="input",
        return_messages=True,
        output_key="output",
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=False,
        return_source_documents=True,
    )

    tools = [
        Tool(
            name="doc_search_tool",
            func=qa,
            description=(
                "This tool is used to retrieve information from the knowledge base"
            ),
        )
    ]

    agent = initialize_agent(
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        tools=tools,
        llm=llm,
        memory=memory,
        return_source_documents=True,
        return_intermediate_steps=True,
        agent_kwargs={"system_message": system_message},
    )

    return agent


def save_uploadedfile(uploadedfile, FILES_PATH):
    with open(os.path.join(FILES_PATH, uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
