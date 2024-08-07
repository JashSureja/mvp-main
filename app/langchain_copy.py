import os
import json
import docx2txt
import psycopg
from flask import jsonify
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import EnsembleRetriever
from langchain_anthropic import AnthropicLLM
from langchain_cohere.llms import Cohere
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import (
    BSHTMLLoader,
    PyMuPDFLoader,
    RecursiveUrlLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.utils import ConfigurableField
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_fireworks import Fireworks
from langchain_google_genai import GoogleGenerativeAI
from langchain_openai import ChatOpenAI, OpenAI
from langchain_groq import ChatGroq
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_models.huggingface import ChatHuggingFace
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_together import ChatTogether, Together
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from urllib.parse import urlparse
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.prompt import SQL_FUNCTIONS_SUFFIX
from langchain_core.messages import AIMessage
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import create_openai_tools_agent
from langchain.agents.agent import AgentExecutor


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return (
            f"Document(page_content={self.page_content!r}, metadata={self.metadata!r})"
        )


class LangChain:
    """
    The LangChain class is responsible for loading the relevant blog,
    processing it, and then generating answers using OpenAI's ChatOpenAI.
    """

    def __init__(self):
        load_dotenv()

        self.prefix_folder = "mvp-documents/"  # for aws folder inside the bucket
        self.bucket = os.getenv("S3_BUCKET_NAME")
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
        print("mongo connected")

        pg_user = os.getenv("POSTGRE_USER_ID")
        pg_password = os.getenv("POSTGRE_PASSWORD")
        pg_host = os.getenv("POSTGRE_HOST")
        pg_port = os.getenv("POSTGRE_PORT")
        pg_dbname = os.getenv("POSTGRE_DBNAME")

        self.db_params = {
            'dbname': pg_dbname,
            'user': pg_user,
            'password': pg_password,
            'host': pg_host,
            'port': pg_port
        }

        # self.db_params = {
        #     "dbname": "steinn_db",
        #     "user": "postgres",
        #     "password": "postgre",
        #     "host": "localhost",
        #     "port": 5432,
        # }

        # self.pg_conn = psycopg.connect(**db_params)
        self.connection_string = (
            os.getenv('DATABASE_URL')
        )
        # self.connection_string = "postgresql://postgres:postgre@localhost/steinn_db"

        #  Get existing files from storage

        self.embeddings = OpenAIEmbeddings()
        #  Create model instance

    def login(self, email, password, organization_id):
        try:
            db_params = self.db_params
            conn = psycopg.connect(**db_params)
            cur = conn.cursor()
            cur.execute(
                "SELECT password FROM users WHERE email = (%s) AND organization_id = (%s)",
                (email, organization_id),
            )
            if password == cur.fetchone()[0]:
                return "success"
            else:
                return "Invalid email or password!"
        except (Exception, psycopg.DatabaseError) as error:
            return f"Error: {error}"

    def insert_user(self, first_name, last_name, email, password, organization_name):
        conn = None
        try:
            db_params = self.db_params
            conn = psycopg.connect(**db_params)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO organizations (name) VALUES (%s) ON CONFLICT (name) DO NOTHING RETURNING id",
                (organization_name,),
            )
            organization_id = cur.fetchone()
            conn.commit()
            if organization_id:
                organization_id = organization_id[0]
            else:
                cur.execute(
                    "SELECT id FROM organizations WHERE name = (%s)",
                    (organization_name,),
                )
                organization_id = cur.fetchone()[0]
            cur.execute(
                """INSERT INTO users (first_name, last_name, email, password, organization_id) 
                VALUES (%s, %s, %s, %s, %s)
            """,
                (first_name, last_name, email, password, organization_id),
            )
            conn.commit()
            cur.close()
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error: {error}")
        finally:
            if conn is not None:
                conn.close()

    def get_organization_id(self, organization_name):
        db_params = self.db_params
        conn = psycopg.connect(**db_params)
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM organizations WHERE name = (%s)", (organization_name,)
        )
        organization_id = cur.fetchone()[0]
        conn.commit()
        if conn is not None:
            conn.close()
        return organization_id

    def create_project(self, organization_id, project_name):
        conn = None
        try:
            db_params = self.db_params
            conn = psycopg.connect(**db_params)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO projects (name,organization_id) VALUES (%s,%s) ON CONFLICT (name) DO NOTHING",
                (
                    project_name,
                    organization_id,
                ),
            )
            conn.commit()
            cur.close()
            return "success"
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error: {error}")
        finally:
            if conn is not None:
                conn.close()

    def get_projects(self, organization_id):
        conn = None
        try:
            db_params = self.db_params
            conn = psycopg.connect(**db_params)
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM projects WHERE organization_id = (%b);",
                (organization_id,),
            )
            project_names = [row[0] for row in cur.fetchall()]
            cur.close()
            return project_names
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error: {error}")
        finally:
            if conn is not None:
                conn.close()

    def save_to_db(self, bot_name, organization_id, project_name, configs):
        conn = None
        try:
            json_configs = json.dumps(configs)
            db_params = self.db_params
            conn = psycopg.connect(**db_params)
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO bots (name,organization_id, project_name, config) VALUES (%s,%s,%s,%s) ON CONFLICT (name, project_name) DO NOTHING",
                (bot_name, organization_id, project_name, json_configs),
            )
            conn.commit()
            cur.close()
            return "success"
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error: {error}")
        finally:
            if conn is not None:
                conn.close()

    def get_bots(self, organization_id, project_name):
        conn = None
        try:
            db_params = self.db_params
            conn = psycopg.connect(**db_params)
            cur = conn.cursor()
            cur.execute(
                "SELECT name FROM bots WHERE organization_id = (%b) AND project_name = (%s);",
                (organization_id, project_name),
            )
            bot_names = [row[0] for row in cur.fetchall()]
            cur.close()
            return bot_names
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error: {error}")
        finally:
            if conn is not None:
                conn.close()

    def get_bot_config(self, organization_id, project_name, bot_name):
        conn = None
        try:
            db_params = self.db_params
            conn = psycopg.connect(**db_params)
            cur = conn.cursor()
            cur.execute(
                "SELECT config FROM bots WHERE organization_id = (%b) AND project_name = (%s) AND name = (%s);",
                (organization_id, project_name, bot_name),
            )
            settings = cur.fetchone()[0]
            cur.close()
            print(settings)
            return settings
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error: {error}")
        finally:
            if conn is not None:
                conn.close()

    def upload_document(self, organization_id, chunk_size, chunk_overlap, file):
        #  upload to postgresql database

        if file:
            try:
                filename = file.filename
                file_content = file.read().decode("utf-8")
            except Exception as e:
                print("Error:", e)

        conn = None
        try:
            db_params = self.db_params
            conn = psycopg.connect(**db_params)
        except Exception as error:
            print(error)

        try:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO documents (organization_id, filename, content, chunk_size, chunk_overlap)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (organization_id, filename, file_content, chunk_size, chunk_overlap),
            )

            conn.commit()
            print(f"File {filename} uploaded successfully.")
            cur.close()
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error: {error}")
        finally:
            if conn is not None:
                conn.close()

    def call_pgvector(self, organization_id, file, chunk_size, chunk_overlap, encoding):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        # for each file, check the extension and select loader accordingly

        files_existing = self.get_file_names(organization_id)
        if files_existing is None:
            files_existing = []
        collection_name = (
            file.filename
            + "[chunk size:"
            + str(chunk_size)
            + ","
            + "chunk overlap:"
            + str(chunk_overlap)
            + "]"
        )
        if collection_name not in files_existing:
            try:
                db_params = self.db_params
                conn = psycopg.connect(**db_params)
                cur = conn.cursor()
                try:
                    file_content = file.read().decode(encoding)
                except UnicodeDecodeError as e:
                    return str(e)

                filename = file.filename

                cur.execute(
                    """INSERT INTO documents (organization_id, filename, content, chunk_size, chunk_overlap) VALUES (%s, %s, %s, %s, %s)""",
                    (
                        organization_id,
                        filename,
                        file_content,
                        chunk_size,
                        chunk_overlap,
                    ),
                )
                conn.commit()
                print(f"File {filename} uploaded successfully.")
                cur.close()
            except Exception as error:
                return str(error)
            finally:
                if conn is not None:
                    conn.close()

            metadata = {"organization_id": organization_id, "source": filename}

            # Combine content and metadata into a document structure
            document = [Document(page_content=file_content, metadata=metadata)]

            documents = text_splitter.split_documents(document)

            collection_name = (
                file.filename
                + "[chunk size:"
                + str(chunk_size)
                + ","
                + "chunk overlap:"
                + str(chunk_overlap)
                + "]"
            )

            PGVector.from_documents(
                documents=documents,
                embedding=self.embeddings,
                connection=self.connection_string,
                collection_name=collection_name,
            )

            return "Vectorstore Creation Successful!"

        else:
            return "Vectorstore Exists!"

    def get_file_names(self, organization_id):
        conn = None
        try:
            db_params = self.db_params
            conn = psycopg.connect(**db_params)
        except Exception as error:
            print(error)

        try:
            cur = conn.cursor()
            cur.execute(
                "SELECT CONCAT(filename, '[chunk size:' , chunk_size , ',' , 'chunk overlap:', chunk_overlap , ']') FROM documents WHERE organization_id = (%b) ORDER BY filename ASC;",
                (organization_id,),
            )
            file_names = [row[0] for row in cur.fetchall()]
            cur.close()
            return file_names
        except (Exception, psycopg.DatabaseError) as error:
            print(f"Error: {error}")
        finally:
            if conn is not None:
                conn.close()

    def connect_vectorstores(self, collection_array):
        retriever_array = []
        #  CREATING RETRIEVER from all retrivers
        try:
            for filename in collection_array:
                vectorstore = PGVector.from_existing_index(
                    embedding=self.embeddings,
                    collection_name=filename,
                    connection=self.connection_string,
                )
                print("VectorStore connected")
                retriever = vectorstore.as_retriever(search_type="similarity")
                retriever_array.append(retriever)

            self.ensemble_retriever = EnsembleRetriever(retrievers=retriever_array)
            return True
        except Exception as error:
            return f"Error: {error}"

    def create_rag_chain(self, settings_params):
        top_k = int(settings_params["top_k"])
        top_p = float(settings_params["top_p"])
        temperature = float(settings_params["temperature"])
        model_name = settings_params["model_name"]
        provider = settings_params["provider"]
        system_prompt = settings_params["system_prompt"]
        self.chat_history = settings_params["chat_history"]

        try:
            if provider == "anthropic":
                llm = AnthropicLLM(
                    model=model_name,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_tokens=128,
                )
            elif provider == "openai":
                llm = ChatOpenAI(
                    model=model_name,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=128,
                )

            elif provider == "cohere":
                # breakpoint()
                llm = Cohere(
                    temperature=temperature,
                    p=top_p,
                    k=top_k,
                    max_tokens=128,
                )

            elif provider == "fireworks":
                llm = Fireworks(
                    model=model_name,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_tokens=1024,
                )
            elif provider == "google":
                llm = GoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_output_tokens=128,
                )
            elif provider == "mistral":
                llm = ChatMistralAI(
                    model_name=model_name,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=128,
                )
            elif "together" in provider:
                llm = ChatTogether(
                    model=model_name,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=128,
                    model_kwargs={"top_k": top_k},
                )
        except Exception as error:
            print(error)

        system_prompt = (
            system_prompt
            + """Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, say that you 
                don't know. Use three sentences maximum and keep the 
                answer concise.
                \n\n
                {context}"""
        )

        if self.chat_history == "on":
            #  Create history aware retriever
            try:
                history_aware_retriever = create_history_aware_retriever(
                    llm, self.ensemble_retriever, self.contextualize_q_prompt
                )
                qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

                rag_chain = create_retrieval_chain(
                    history_aware_retriever, question_answer_chain
                )
            except Exception as error:
                print(error)

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
            # Create normal retriever
            prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("human", "{input}")]
            )
            try:
                question_answer_chain = create_stuff_documents_chain(llm, prompt)
                self.rag_chain = create_retrieval_chain(
                    self.ensemble_retriever, question_answer_chain
                )
                print("rag chain built successfully")
            except Exception as error:
                print(error)

    def call_html_parser(self, organization_id, project_id, url, settings_params):
        collection_name = organization_id + project_id + url

        top_k = settings_params["top_k"]
        top_p = settings_params["top_p"]
        temperature = settings_params["temperature"]
        model_name = settings_params["model_name"]
        provider = settings_params["provider"]

        if provider == "anthropic":
            llm = AnthropicLLM(
                model=model_name,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=128,
            )
        elif provider == "openai":
            llm = OpenAI(
                model=model_name, temperature=temperature, top_p=top_p, max_tokens=128
            )

        elif provider == "cohere":
            llm = Cohere(temperature=temperature, p=top_p, k=top_k, max_tokens=128)

        elif provider == "fireworks":
            llm = Fireworks(
                model=model_name,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                max_tokens=1024,
            )
        elif provider == "google":
            llm = (
                GoogleGenerativeAI(
                    model=model_name,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_output_tokens=128,
                ),
            )

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
        retriever = web_vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": top_k}
        )
        system_prompt = """Use the following pieces of retrieved context to answer the question. 
                If you don't know the answer, say that you 
                don't know. Use three sentences maximum and keep the 
                answer concise.\n\n
                {context}"""

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        return rag_chain

    # def call_csv_agent(query):
    #     csv_file = "uploaded csv file"

    #     if csv_file is not None:
    #         agent = create_csv_agent(OpenAI(temperature=0), csv_file, verbose=True)

    #         user_question = query

    #         if user_question is not None and user_question != "":
    #             agent.invoke(user_question)

    def get_tokens(self, sentence):
        array = sentence.split()
        return len(array) / 0.75

    def sql_agent(self):
        db = self.db

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        toolkit = SQLDatabaseToolkit(db=db, llm=llm)
        context = toolkit.get_context()
        tools = toolkit.get_tools()

        messages = [
            HumanMessagePromptTemplate.from_template("{input}"),
            AIMessage(content=SQL_FUNCTIONS_SUFFIX),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        prompt = prompt.partial(**context)

        agent = create_openai_tools_agent(llm, tools, prompt)

        agent_executor = AgentExecutor(
            agent=agent,
            tools=toolkit.get_tools(),
            verbose=True,
        )

        return agent_executor

    def get_response(self, query):
        if self.chat_history == "on":
            try:
                response = self.conversational_rag_chain.invoke(
                    {"input": query},
                    config={
                        "configurable": {"session_id": "abc123"}
                    },  # constructs a key "abc123" in `store`.
                )
            except Exception as error:
                response = error
        else:
            try:
                response = self.rag_chain.invoke({"input": query})
            except Exception as error:
                response = error
        print(response)
        return response
