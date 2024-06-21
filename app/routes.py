
import os
import os.path

from flask import Blueprint, redirect, render_template, request, url_for, session, current_app, jsonify
from .extensions import db
from .models import User


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

@main.route('/')
def upload_page():
    return render_template('upload.html')


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
    if len(documents) != 0:
        langchain.connect_vectorstores(1,1,documents, settings_params)
        return render_template('index.html', settings=settings_params, message="Settings saved successfully!", documents=files_existing)
    else: 
        return render_template('index.html', settings=settings_params, message="Select at least one document.", documents=files_existing)

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
    # chunk_size = request.form.get('chunk_size')
    # chunk_overlap = request.form.get('chunk_overlap')
    chunk_size = 2000
    chunk_overlap = 200
    for file in files:
        if file.filename != '':
            settings_params['documents'].append(file.filename)
            
            message = langchain.call_pgvector(1, 1, file, chunk_size, chunk_overlap)
            # langchain.upload_files("org_1", "project_1", file)
            
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

