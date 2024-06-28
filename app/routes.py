
import os
import os.path

from urllib.parse import urlparse
from flask import Blueprint, redirect, render_template, request, url_for, session, current_app, jsonify
from .extensions import db
from .models import User
from langchain_community.utilities import SQLDatabase

main = Blueprint('main', __name__)
settings_params = {
    'provider': '',
    'model_name':  '',
    'temperature': 0.4,
    'top_p': 0.7,
    'top_k': 3,
    'system_prompt': '',
    'documents': [],
    'existing_docs':[]
}

@main.route("/")
def home():
    if 'logged_in' in session and session['logged_in']:
        return render_template('menu.html')
    return render_template('user_login.html')



@main.route('/signup', methods=['GET'])
def signup():
    return render_template('signup.html')


@main.route('/login', methods=['GET'])
def login():
    return render_template('login.html')


@main.route('/signup_form', methods=['POST'])
def signup_form():
    langchain = current_app.config["LANGCHAIN"]
    data = request.form
    first_name = data['first_name']
    last_name = data['last_name']
    org_name = data['org_name']
    email = data['email']
    password = data['password']
    langchain.insert_user(first_name, last_name, org_name, email, password)
    return render_template('menu.html')


@main.route('/login_form', methods=['POST'])
def login_form():
    password = request.form['password']
    if password == os.getenv('SESSION_PASSWORD'):
        session['logged_in'] = True
        return render_template('menu.html')
    return render_template('login.html', error="Invalid password")


@main.route('/menu', methods=['POST'])
def menu():
    langchain = current_app.config["LANGCHAIN"]
    existing_documents = langchain.get_file_names()
    selected_menu = request.form['selected_menu']
    if selected_menu == "upload_documents":
        return render_template('index.html',settings=settings_params, message="", documents=existing_documents)

    # elif selected_menu == "website_agent":
    #     return render_template('web_agent.html',settings=settings_params, message="")

    elif selected_menu == "sql_agent":
        return render_template('sql_agent.html',settings=settings_params, message="")


@main.route('/sql_set_uri', methods=['POST'])
def save_uri():
    langchain = current_app.config["LANGCHAIN"]
    data = request.form
    connection_string = str(data.get('connection_string'))
    
    langchain.db = SQLDatabase.from_uri(connection_string)
    langchain.agent = langchain.sql_agent()
    return render_template('sql_agent.html', message="Database Connected", connection_string = connection_string)


@main.route('/sql_input', methods=['GET', 'POST'])
def sql_process():
    langchain = current_app.config["LANGCHAIN"]
    data = request.form
    connection_string = str(data.get('connection_string'))
    query = data.get('message')
    response = langchain.agent.invoke({"input": query})
    return render_template('sql_agent.html', message=response['output'], connection_string = connection_string)


@main.route('/index')
def index():
    langchain = current_app.config["LANGCHAIN"]
    existing_documents = langchain.get_file_names()
    return render_template('index.html',settings=settings_params, message="", documents=existing_documents)



# @main.route('/documents', methods=['POST','GET'])
# def get_docs():
#     global settings_params
#     data = request.form
#     selected_docs = data.getlist('documents')
#     settings_params = { 'documents' : selected_docs}


@main.route('/settings', methods=['POST','GET'])
def chat():
    
    
    global settings_params
    
    langchain = current_app.config["LANGCHAIN"]
    files_existing = langchain.get_file_names()
    data = request.form

    settings_params = {
        'provider': data.get('provider', ''),
        'model_name': data.get('model_name', ''),
        'temperature': data.get('temperature', 0.4),
        'top_p': data.get('top_p', 0.7),
        'top_k': data.get('top_k', 3),
        'system_prompt': data.get('system_prompt', ''),
        'cite_sources': data.get('cite_sources', ''),
        'chat_history': data.get('chat_history',''),
        'documents' : data.getlist('documents'),
        'existing_docs' : files_existing
    }
    print(settings_params)
    documents = settings_params['documents']
    if len(documents) != 0 :
        langchain.connect_vectorstores(1,1,documents, settings_params)
        return render_template('index.html', settings=settings_params, message="Settings saved successfully!", documents=files_existing)
    else: 
        return render_template('index.html', settings=settings_params, message="Please select at least one document for retrieval.", documents=files_existing)


@main.route('/send_message', methods=['POST'])
def send_message():
    
    langchain = current_app.config["LANGCHAIN"]
    global settings_params
    message = request.form.get('message')
    cite_sources = settings_params.get('cite_sources')
    response = langchain.get_response(message)
    if cite_sources == "on":
        sources = ["Sources:-"]
        for doc in response['context']:
            sources.append(doc.metadata['source'])
        
        
        return render_template('index.html', settings=settings_params, message=response['answer'], sources=sources, documents=settings_params['existing_docs'])
    else:    
        return render_template('index.html', settings=settings_params, message=response["answer"], sources='', documents=settings_params['existing_docs'])


@main.route('/upload', methods=['POST'])
def upload_file():
    
    langchain = current_app.config["LANGCHAIN"]
    global settings_params
    files = request.files.getlist('documents')
    message = ""
    chunk_size = request.form.get('chunk_size')
    chunk_overlap = request.form.get('chunk_overlap')
    # chunk_size = 2000
    # chunk_overlap = 200
    for file in files:
        if file.filename != '':
            settings_params['documents'].append(file.filename)
            
            message = langchain.call_pgvector(1, 1, file, int(chunk_size), int(chunk_overlap))
            # langchain.upload_files("org_1", "project_1", file)
            # print(chunk_overlap,chunk_size)
            langchain.upload_document(1, 1,file)
            
            
    files_existing = langchain.get_file_names()
    if message != "":
        pass
    else:
        message = "Select a file for upload!"
    return render_template('index.html', settings=settings_params, message=message, documents=files_existing)


@main.route('/add/<username>')
def add_user(username):
    db.session.add(User(username=username))
    db.session.commit()
    return redirect(url_for("main.index"))

