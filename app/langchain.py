import os

# import docx2txt
import psycopg
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import (
    BSHTMLLoader,
    PyMuPDFLoader,
    RecursiveUrlLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_openai import OpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LangChain:
    """
    The LangChain class is responsible for loading the relevant blog,
    processing it, and then generating answers using OpenAI's ChatOpenAI.
    """

    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        self.template = """You are assisting a hormone therapy doctor. Use the following pieces of transcript to answer the question at the end.
        If you don't know the answer or if the answer does not belong in the context, just say that you don't know, don't try to make up an answer.

        {context}

        """

        mongo_user = os.getenv("MONGO_USER_ID")
        mongo_password = os.getenv("MONGO_PASSWORD")
        mongo_cluster_uri = os.getenv("MONGO_CLUSTER_URI")
        self.uri = (
            "mongodb+srv://"
            + mongo_user
            + ":"
            + mongo_password
            + "@"
            + mongo_cluster_uri
        )

        pg_user = os.getenv("POSTGRE_USER_ID")
        pg_password = os.getenv("POSTGRE_PASSWORD")
        self.embeddings = OpenAIEmbeddings()
        self.connection_string = (
            "postgresql+psycopg://"
            + pg_user
            + ":"
            + pg_password
            + "@localhost:5432/steinn_db"
        )

    def call_csv_agent(query):
        csv_file = "uploaded csv file"

        if csv_file is not None:
            agent = create_csv_agent(OpenAI(temperature=0), csv_file, verbose=True)

            user_question = query

            if user_question is not None and user_question != "":
                agent.invoke(user_question)

    def call_pgvector(
        self, organisation_id, project_id, files, chunk_size, chunk_overlap
    ):
        # files array contains file names

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        files = os.listdir("uploads/")
        # for each file, check the extension and select loader accordingly
        for file in files:
            
            collection_name = organisation_id + project_id + file

            if file.endswith(".txt"):
                loader = TextLoader(os.path.join('uploads/', file))
                doc = loader.load()

            elif file.endswith(".pdf"):
                loader = PyMuPDFLoader(os.path.join('uploads/', file))
                doc = loader.load()

            # elif file.filename.endswith('.docx'):
            #     doc = docx2txt.process(file)

            else:
                return "Not a valid file format"

            documents = text_splitter.split_documents(doc)
            vectorstore = PGVector.from_documents(
                documents=documents,
                embedding=self.embeddings,
                connection=self.connection_string,
                collection_name=collection_name,
            )

    def call_html_parser(self, url):
        collection_name = "organisation_id" + "project_id" + url

        loader = RecursiveUrlLoader(url, prevent_outside=True)
        html2text = Html2TextTransformer()
        doc = loader.load()
        # filter docs for only text content
        filtered_docs = [
            d
            for d in doc
            if "content_type" in d.metadata and "text" in d.metadata["content_type"]
        ]

        docs_transformed = html2text.transform_documents(filtered_docs)

        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(docs_transformed)
        print("splitted")

        web_vectorstore = PGVector.from_documents(
            documents=documents,
            embedding=self.embeddings,
            connection=self.connection_string,
            collection_name=collection_name,
        )
        print("vector done")

    def connect_vectorstores(self, collection_array, top_k):
        retriever_array = []
        # collection_array to be created as per user gives the docs to be included in the bot
        for filename in collection_array:
            vectorstore = PGVector.from_existing_index(
                embedding=self.embeddings,
                collection_name= "org_1project_1" +filename,
                connection=self.connection_string,
            )
            print("VectorStore connected")
            retriever = vectorstore.as_retriever(
                search_kwargs={"k": top_k},
            )
            retriever_array.append(retriever)

        ensemble_retriever = EnsembleRetriever(
            retrievers=retriever_array
        )
        print("Retriever created")
        return vectorstore
