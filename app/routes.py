import os
import os.path
from flask import (
    Blueprint,
    redirect,
    render_template,
    request,
    url_for,
    session,
    current_app,
    flash,
)
from .extensions import db
from .models import User
from langchain_community.utilities import SQLDatabase

main = Blueprint("main", __name__)
settings_params = {
    "provider": "",
    "model_name": "",
    "temperature": 0.4,
    "top_p": 0.7,
    "top_k": 3,
    "system_prompt": "",
    "documents": [],
    "existing_docs": [],
    "chunk_size": 1000,
    "chunk_overlap": 50,
}

settings_store = []  # for multi page
selected_documents = []  # for multi page


@main.route("/")
def home():
    if "logged_in" in session and session["logged_in"]:
        return render_template("user_login.html")
    return render_template("user_login.html")


@main.route("/signup", methods=["GET"])
def signup():
    return render_template("signup.html")


@main.route("/login", methods=["GET"])
def login():
    return render_template("login.html")


@main.route("/signup_form", methods=["POST"])
def signup_form():
    langchain = current_app.config["LANGCHAIN"]
    data = request.form
    first_name = data["first_name"]
    last_name = data["last_name"]
    org_name = data["org_name"]
    email = data["email"]
    password = data["password"]
    langchain.insert_user(first_name, last_name, email, password, org_name)
    global organization_id
    organization_id = langchain.get_organization_id(org_name)
    projects = langchain.get_projects(organization_id)
    return render_template("login.html", projects=projects)


@main.route("/login_form", methods=["POST"])
def login_form():
    global organization_id
    langchain = current_app.config["LANGCHAIN"]
    langchain = current_app.config["LANGCHAIN"]
    data = request.form
    org_name = data["org_name"]
    email = data["email"]
    password = data["password"]
    organization_id = langchain.get_organization_id(org_name)
    message = langchain.login(email, password, organization_id)
    if message == "success":
        projects = langchain.get_projects(organization_id)
        return render_template("project_menu.html", projects=projects)
    else:
        return render_template("login.html", error=message)


@main.route("/create", methods=["POST"])
def create():
    global organization_id, project_name
    langchain = current_app.config["LANGCHAIN"]
    project_name = request.form["project_name"]
    langchain.create_project(organization_id, project_name)
    return render_template("menu.html")


@main.route("/create_bot", methods=["POST"])
def menu():
    global organization_id
    langchain = current_app.config["LANGCHAIN"]
    existing_documents = langchain.get_file_names(organization_id)
    selected_menu = request.form["selected_menu"]
    if selected_menu == "rag_agent":
        if existing_documents is not None:
            return render_template(
                "index.html",
                settings=settings_params,
                message="",
                documents=existing_documents,
            )
        else:
            return render_template(
                "index.html", settings=settings_params, message="", documents=[]
            )
    elif selected_menu == "multi_agent":
        return render_template(
            "index_copy.html",
            settings_store=settings_store,
            selected_documents=[],
            documents=existing_documents,
        )

    elif selected_menu == "sql_agent":
        return render_template("sql_agent.html", settings=settings_params, message="")


@main.route("/select_project", methods=["GET", "POST"])
def select_project():
    global organization_id, project_name
    langchain = current_app.config["LANGCHAIN"]
    project_name = request.form["selected_project"]
    bot_names = langchain.get_bots(organization_id, project_name)
    return render_template("bot_names.html", project=project_name, bots=bot_names)


@main.route("/load_bot", methods=["POST"])
def load_bot():
    global organization_id, project_name
    langchain = current_app.config["LANGCHAIN"]
    bot_name = request.form["selected_bot"]
    settings_params = langchain.get_bot_config(organization_id, project_name, bot_name)
    existing_documents = langchain.get_file_names(organization_id)
    return render_template(
        "index.html", settings=settings_params, message="", documents=existing_documents
    )


@main.route("/sql_set_uri", methods=["POST"])
def save_uri():
    langchain = current_app.config["LANGCHAIN"]
    data = request.form
    connection_string = str(data.get("connection_string"))

    langchain.db = SQLDatabase.from_uri(connection_string)
    langchain.agent = langchain.sql_agent()
    return render_template(
        "sql_agent.html",
        message="Database Connected",
        connection_string=connection_string,
    )


@main.route("/sql_input", methods=["GET", "POST"])
def sql_process():
    langchain = current_app.config["LANGCHAIN"]
    data = request.form
    connection_string = str(data.get("connection_string"))
    query = data.get("message")
    response = langchain.agent.invoke({"input": query})
    return render_template(
        "sql_agent.html",
        message=response["output"],
        connection_string=connection_string,
    )


@main.route("/index", methods=["GET", "POST"])
def index():
    global organization_id
    return render_template("menu.html")


# @main.route('/documents', methods=['POST','GET'])
# def get_docs():
#     global settings_params
#     data = request.form
#     selected_docs = data.getlist('documents')
#     settings_params = { 'documents' : selected_docs}


@main.route("/settings", methods=["POST", "GET"])
def chat():
    global settings_params, organization_id

    langchain = current_app.config["LANGCHAIN"]
    files_existing = langchain.get_file_names(organization_id)
    data = request.form

    settings_params = {
        "provider": data.get("provider", ""),
        "model_name": data.get("model_name", ""),
        "temperature": data.get("temperature", 0.4),
        "top_p": data.get("top_p", 0.7),
        "top_k": data.get("top_k", 3),
        "system_prompt": data.get("system_prompt", ""),
        "cite_sources": data.get("cite_sources", ""),
        "chat_history": data.get("chat_history", ""),
        "documents": data.getlist("documents"),
        "existing_docs": files_existing,
    }
    print(settings_params)
    documents = settings_params["documents"]
    if len(documents) != 0:
        retriever_created = langchain.connect_vectorstores(documents)
        if retriever_created:
            langchain.create_rag_chain(settings_params)
            return render_template(
                "index.html",
                settings=settings_params,
                message="Settings saved successfully!",
                documents=files_existing,
            )
    else:
        flash("Please select at least one document for retrieval.")
        return render_template(
            "index.html", settings=settings_params, documents=files_existing
        )


@main.route("/send_message", methods=["POST"])
def send_message():
    langchain = current_app.config["LANGCHAIN"]
    global settings_params
    message = request.form.get("message")
    cite_sources = settings_params.get("cite_sources")
    response = langchain.get_response(message)
    if cite_sources == "on":
        sources = ["Sources:-"]
        for doc in response["context"]:
            sources.append(doc.metadata["source"])
        return render_template(
            "index.html",
            settings=settings_params,
            message=response["answer"],
            sources=sources,
            documents=settings_params["existing_docs"],
        )
    else:
        return render_template(
            "index.html",
            settings=settings_params,
            message=response["answer"],
            sources="",
            documents=settings_params["existing_docs"],
        )


@main.route("/upload", methods=["POST"])
def upload_file():
    langchain = current_app.config["LANGCHAIN"]
    global settings_params, organization_id
    files = request.files.getlist("documents")
    message = ""
    chunk_size = request.form.get("chunk_size")
    chunk_overlap = request.form.get("chunk_overlap")
    encoding = request.form.get("encoding")
    if encoding == "":
        encoding = "utf-8"
    if chunk_size == "":
        chunk_size = 2000
    if chunk_overlap == "":
        chunk_overlap = 200
    # chunk_size = 2000
    # chunk_overlap = 200
    print("in file upload fn")
    for file in files:
        if file.filename != "":
            print("in if loop")

            message = langchain.call_pgvector(
                organization_id, file, int(chunk_size), int(chunk_overlap), encoding
            )

    print("outside if")
    files_existing = langchain.get_file_names(organization_id)
    if "Vectorstore" in message:
        settings_params["documents"].append(file.filename)
        return render_template(
            "index.html",
            settings=settings_params,
            message="Vector store created",
            documents=files_existing,
        )

    else:
        flash(message, "error")
        return render_template(
            "index.html",
            settings=settings_params,
            message=message,
            documents=files_existing,
        )


@main.route("/save_to_db", methods=["POST"])
def save_to_db():
    global organization_id, project_name, settings_params
    langchain = current_app.config["LANGCHAIN"]
    files_existing = langchain.get_file_names(organization_id)
    print(settings_params)
    bot_name = request.form.get("bot_name", "")
    langchain.save_to_db(bot_name, organization_id, project_name, settings_params)
    return render_template(
        "index.html",
        settings=settings_params,
        message="Saved to DB",
        documents=files_existing,
    )


@main.route("/add/<username>")
def add_user(username):
    db.session.add(User(username=username))
    db.session.commit()
    return redirect(url_for("main.index"))


# multi page routes


@main.route("/multi", methods=["GET"])
def login_multi():
    global settings_store, organization_id
    organization_id = 22
    langchain = current_app.config["LANGCHAIN"]
    files_existing = langchain.get_file_names(organization_id)

    # settings_store = jsonify(settings_store)
    return render_template(
        "index_copy.html",
        settings_store=settings_store,
        selected_documents=[],
        documents=files_existing,
    )


@main.route("/settings/<int:block_id>", methods=["POST", "GET"])
def chat_multi(block_id):
    global organization_id, settings_store, selected_documents
    organization_id = 22
    langchain = current_app.config["LANGCHAIN"]
    files_existing = langchain.get_file_names(organization_id)

    settings = {
        "block_id": block_id,
        "provider": request.form.get(f"provider-{block_id}"),
        "model_name": request.form.get(f"model_name-{block_id}"),
        "temperature": request.form.get("temperature", 0.4),
        "top_p": request.form.get("top_p", 0.7),
        "top_k": request.form.get("top_k", 3),
        "cite_sources": request.form.get("cite_sources"),
        "chat_history": request.form.get("chat_history"),
        "system_prompt": request.form.get("system_prompt", ""),
        "documents": selected_documents,
        "query": "",
        "answer": "",
        "sources": "",
        "output_tokens": "",
        "input_tokens": "",
    }
    while len(settings_store) < block_id + 1:
        settings_store.append(
            {
                "block_id": block_id,
                "provider": "",
                "model_name": "",
                "temperature": 0.4,
                "top_p": 0.7,
                "top_k": 2,
                "cite_sources": "",
                "chat_history": "",
                "system_prompt": "",
                "documents": "",
                "query": "",
                "answer": "",
                "sources": "",
                "output_tokens": "",
                "input_tokens": "",
            }
        )
    settings_store[block_id] = settings
    print(settings_store)
    cite_sources = settings.get("cite_sources")
    if len(selected_documents) != 0:
        query = request.form.get(f"message-{block_id}")
        print(query)
        try:
            langchain.create_rag_chain(settings)
        except Exception as error:
            settings_store[block_id].update({"answer": error})
            return render_template(
                "index_copy.html",
                settings_store=settings_store,
                selected_documents=selected_documents,
                documents=files_existing,
            )
        response = langchain.get_response(query)
        settings_store[block_id].update({"query": "Input : " + str(query)})
        input_tokens = langchain.get_tokens(query)
        settings_store[block_id].update(
            {"input_tokens": "Input tokens : " + str(input_tokens)}
        )
        try:
            llm_answer = response.get("answer")
            output_tokens = langchain.get_tokens(llm_answer)
            settings_store[block_id].update(
                {"output_tokens": "Output tokens : " + str(output_tokens)}
            )
            settings_store[block_id].update({"answer": "Output : " + llm_answer})

            if cite_sources == "on":
                sources = ["Sources :-"]
                for doc in response["context"]:
                    sources.append(doc.metadata["source"])
                settings_store[block_id].update({"sources": sources})
                return render_template(
                    "index_copy.html",
                    settings_store=settings_store,
                    selected_documents=selected_documents,
                    documents=files_existing,
                )
            else:
                return render_template(
                    "index_copy.html",
                    settings_store=settings_store,
                    selected_documents=selected_documents,
                    documents=files_existing,
                )
        except Exception:
            flash(str(response))
            return render_template(
                "index_copy.html",
                settings_store=settings_store,
                selected_documents=selected_documents,
                documents=files_existing,
            )

    else:
        flash("Select at least one document for retrieval.")
        return render_template(
            "index_copy.html",
            settings_store=settings_store,
            selected_documents=selected_documents,
            documents=files_existing,
        )


@main.route("/close_block/<int:block_id>", methods=["POST"])
def close_block(block_id):
    global settings_store, selected_documents
    langchain = current_app.config["LANGCHAIN"]
    files_existing = langchain.get_file_names(organization_id)
    try:
        settings_store.pop(block_id)
    except Exception as error:
        return render_template(
            "index_copy.html",
            settings_store=settings_store,
            selected_documents=selected_documents,
            documents=files_existing,
        )

    return render_template(
        "index_copy.html",
        settings_store=settings_store,
        selected_documents=selected_documents,
        documents=files_existing,
    )


@main.route("/upload_multi", methods=["POST"])
def upload_multi():
    langchain = current_app.config["LANGCHAIN"]
    files_existing = langchain.get_file_names(organization_id)
    files = request.files.getlist("documents")
    chunk_size = request.form.get("chunk_size", "2000")
    chunk_overlap = request.form.get("chunk_overlap", "200")
    encoding = request.form.get("encoding")
    if encoding == "":
        encoding = "utf-8"
    if chunk_size == "":
        chunk_size = 2000
    if chunk_overlap == "":
        chunk_overlap = 200
    print(files)
    for file in files:
        if file.filename != "":
            print("in if loop")
            selected_documents.append(file.filename)

            response = langchain.call_pgvector(
                organization_id, file, int(chunk_size), int(chunk_overlap), encoding
            )
            if "Vectorstore" in response:
                files_existing = langchain.get_file_names(organization_id)
                return render_template(
                    "index_copy.html",
                    settings_store=settings_store,
                    selected_documents=selected_documents,
                    documents=files_existing,
                )

            else:
                flash(response, "error")
                return render_template(
                    "index_copy.html",
                    settings_store=settings_store,
                    selected_documents=selected_documents,
                    documents=files_existing,
                )
        else:
            flash("Select a file with a proper name!")
            return render_template(
                "index_copy.html",
                settings_store=settings_store,
                selected_documents=selected_documents,
                documents=files_existing,
            )


@main.route("/docs_selected", methods=["POST"])
def docs_selected():
    global selected_documents
    langchain = current_app.config["LANGCHAIN"]
    files_existing = langchain.get_file_names(organization_id)
    selected_documents = request.form.getlist("documents")
    if len(selected_documents) != 0:
        retriever_created = langchain.connect_vectorstores(selected_documents)
        if retriever_created:
            return render_template(
                "index_copy.html",
                settings_store=settings_store,
                selected_documents=selected_documents,
                documents=files_existing,
            )
    else:
        flash("Select at least one document for retrieval.")
        return render_template(
            "index_copy.html",
            settings_store=settings_store,
            selected_documents=selected_documents,
            documents=files_existing,
        )


@main.route("/save_to_db/<int:block_id>", methods=["POST"])
def save_to_db_multi(block_id):
    global organization_id, project_name, settings_params
    langchain = current_app.config["LANGCHAIN"]
    files_existing = langchain.get_file_names(organization_id)
    print(settings_store[block_id])
    bot_name = request.form.get("bot_name", "")
    langchain.save_to_db(
        bot_name, organization_id, project_name, settings_store[block_id]
    )
    settings_store[block_id].update({"answer": "Saved In The Database"})
    return render_template(
        "index_copy.html",
        settings_store=settings_store,
        selected_documents=selected_documents,
        documents=files_existing,
    )
