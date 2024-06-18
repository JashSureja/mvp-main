
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
    'documents': []
}

@main.route('/')
def upload_page():
    return render_template('upload.html')


@main.route('/index')
def index():
    documents = settings_params['documents']
    return render_template('index.html',settings=settings_params, message="", documents=documents)


    


@main.route('/settings', methods=['POST','GET'])
def chat():
    
    global settings_params
    langchain = current_app.config["LANGCHAIN"]
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
        'documents': data.getlist('documents')
    }
    print(settings_params)
    documents=settings_params['documents']
    langchain.connect_vectorstores(documents, settings_params)

    return render_template('index.html', settings=settings_params, message="Settings saved successfully!", documents=documents)

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
        
        
        return render_template('index.html', settings=settings_params, message=response['answer'], sources=sources, documents=settings_params['documents'])
    else:    
        return render_template('index.html', settings=settings_params, message=response["answer"], sources='', documents=settings_params['documents'])


@main.route('/upload', methods=['POST'])
def upload_file():
    
    langchain = current_app.config["LANGCHAIN"]
    global settings_params
    files = request.files.getlist('documents')
    # chunk_size = request.form.get('chunk_size')
    # chunk_overlap = request.form.get('chunk_overlap')
    chunk_size = 2000
    chunk_overlap = 200
    for file in files:
        if file.filename != '':
            settings_params['documents'].append(file.filename)
            file.save(os.path.join(current_app.config['UPLOAD_FOLDER'], file.filename))
            langchain.call_pgvector("org_1", "project_1", file, chunk_size, chunk_overlap)



    
    return redirect(url_for('main.index'))

@main.route('/add/<username>')
def add_user(username):
    db.session.add(User(username=username))
    db.session.commit()
    return redirect(url_for("main.index"))

