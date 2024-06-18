import os

import docx2txt
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
from langchain_openai import OpenAI, ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_anthropic import ChatAnthropic
from langchain_core.runnables.utils import ConfigurableField
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_cohere import ChatCohere
from langchain_fireworks import Fireworks
from langchain_google_genai import GoogleGenerativeAI

class LangChain:
    """
    The LangChain class is responsible for loading the relevant blog,
    processing it, and then generating answers using OpenAI's ChatOpenAI.
    """

    def __init__(self):
        load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_key =  os.getenv("ANTHROPIC_API_KEY")
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        self.fireworks_api_key = os.getenv("FIREWORKS_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        self.contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # self.template = """You are assisting a hormone therapy doctor. Use the following pieces of transcript to answer the question at the end.
        # If you don't know the answer or if the answer does not belong in the context, just say that you don't know, don't try to make up an answer.

        # {context}

        # """
        
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

#  Create model instance
        self.llm = ChatAnthropic(
            model="claude-3-haiku-20240307", temperature=0, top_k=3, top_p=0.7, max_tokens=1024
        ).configurable_alternatives(
            ConfigurableField(id="provider"),
            default_key="anthropic",
            openai=ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                max_tokens=128
            ),
            cohere = ChatCohere(
                model="command-r",
                temperature=0,
                max_tokens=128
            ),
            fireworks = Fireworks(
                model="accounts/fireworks/models/firefunction-v2",
                temperature=0,
                top_k = 3,
                top_p=0.7,
                max_tokens=128
            ),
            google = GoogleGenerativeAI(
                model = "text-bison@002",
                temperature=0,
                top_k = 3,
                top_p=0.7,
                max_output_tokens=128

            )
            
            # anthropic=AnthropicLLM(
            #     model="claude-3-haiku-20240307",
            #     temperature=0,
            #     top_p=0.7,
            #     top_k = 3
            # ),
            # openai=OpenAI(
            #     model="gpt-4o",
            #     temperature=0,
            #     top_p=0.7,
            #     top_k = 3
            # ),
            # cohere = Cohere(
            #     model="command-r",
            #     temperature=0,
            #     top_p=0.7,
            #     top_k = 3
            # ),
           


        ).configurable_fields(
            model = ConfigurableField(
                id="model_name"
            ),
            temperature=ConfigurableField(
                id="temperature",
                name="Temperature of the llm",
                description="The temperature defines balance between creativity and accuracy of the llm, 1 = creative, 0 = accurate",
            ),
            top_k = ConfigurableField(
                id = "top_k",
                name = "Top k number of documents",
                description="Defines the number of documents to be retrieved"
            ),
            top_p = ConfigurableField(
                id="top_p",
                name = "The top probability",
                description = " Retrieves the documents on the basis of probability value set, may use the sum of probability in case of multiple retrievals"
            ),
            max_tokens = ConfigurableField(
                id = "max_tokens",
                name = "The maximum number of tokens",
                description= "The maximum number of output tokens produced by the model"
            ),
            
        )


    def call_csv_agent(query):
        csv_file = "uploaded csv file"

        if csv_file is not None:
            agent = create_csv_agent(OpenAI(temperature=0), csv_file, verbose=True)

            user_question = query

            if user_question is not None and user_question != "":
                agent.invoke(user_question)

    def call_pgvector(
        self, organisation_id, project_id, file, chunk_size, chunk_overlap
    ):
        # files array contains file names
        vectors = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        files_existing = os.listdir("uploads/")
        # for each file, check the extension and select loader accordingly
        # for file in files:
        #     if file.filename not in files_existing:
        collection_name = organisation_id + project_id + file.filename
        print("inside for loop")
        if ".txt" in file.filename :
            loader = TextLoader(os.path.join('uploads/', file.filename))
            doc = loader.load()

        elif ".pdf" in file.filename :
            loader = PyMuPDFLoader(os.path.join('uploads/', file.filename))
            doc = loader.load()

        elif '.docx' in file.filename :
            doc = docx2txt.process(file)

        else:
            print( "Not a valid file format")

        documents = text_splitter.split_documents(doc)
        vectorstore = PGVector.from_documents(
            documents=documents,
            embedding=self.embeddings,
            connection=self.connection_string,
            collection_name=collection_name,
        )
        #         if vectorstore is not None:
        #             vectors = vectors.append("created")
        # if len(vectors)==len(files):
        print("vectorstore creation successful")

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

    def connect_vectorstores(self, collection_array, settings_params):
        retriever_array = []

        top_k = settings_params["top_k"]
        top_p = settings_params["top_p"]
        temperature = settings_params["temperature"]
        model_name = settings_params["model_name"]
        provider = settings_params["provider"]
        system_prompt = settings_params["system_prompt"]
        self.chat_history = settings_params["chat_history"]
        

#  CREATING RETRIEVER from all retrivers

        for filename in collection_array:
            vectorstore = PGVector.from_existing_index(
                embedding=self.embeddings,
                collection_name= "org_1"+"project_1" +filename,
                connection=self.connection_string,
            )
            print("VectorStore connected")
            retriever = vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k":top_k}
            )
            retriever_array.append(retriever)

        ensemble_retriever = EnsembleRetriever(
            retrievers=retriever_array
        )
        print("Retriever created")
        

    # Create model as per the user request 

        llm = self.llm.with_config(configurable = {
                "provider":provider, 
                "model_name":model_name, 
                "temperature":temperature, 
                "top_k":top_k, 
                "top_p":top_p,
                # "max_tokens":max_tokens
                }
            )
        system_prompt = system_prompt + """Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, say that you 
                don't know. Use three sentences maximum and keep the 
                answer concise.
                \n\n
                {context}"""
        
        if self.chat_history == "on":
            
            #  Create history aware retriever and define system prompt

            history_aware_retriever = create_history_aware_retriever(
                llm, ensemble_retriever, self.contextualize_q_prompt
            )
            
            
            qa_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder("chat_history"),
                    ("human", "{input}"),
                ]
            )
            question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

            rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

            store = {}

            def get_session_history(session_id: str) -> BaseChatMessageHistory:
                if session_id not in store:
                    store[session_id] = ChatMessageHistory()
                return store[session_id]

            self.conversational_rag_chain = RunnableWithMessageHistory(
                rag_chain,
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer",
            )
            print("chat conversation chain built sucessfully")

        else: 

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input}"),
                ]
            )

            question_answer_chain = create_stuff_documents_chain(llm, prompt)
            self.rag_chain = create_retrieval_chain(ensemble_retriever, question_answer_chain)
            print("rag chain built successfully")
            



    def get_response(self, query):
        
        
        if self.chat_history == "on":
            response = self.conversational_rag_chain.invoke(
                {"input": query},
                config={
                    "configurable": {"session_id": "abc123"}
                },  # constructs a key "abc123" in `store`.
            )
        else:
            response = self.rag_chain.invoke({"input": query})

        print(response)
        return response
