
import os
import os.path

from flask import Blueprint, redirect, render_template, request, url_for, session, current_app, jsonify
from .extensions import db
from .models import User


main = Blueprint('main', __name__)
settings_params = {
    'model': '',
    'temperature': 0.4,
    'top_p': 0.7,
    'top_k': 0.0,
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
    # langchain = current_app.config["LANGCHAIN"]
    data = request.form
    settings_params = {
        'model': data.get('model', ''),
        'temperature': data.get('temperature', 0.4),
        'top_p': data.get('top_p', 0.7),
        'top_k': data.get('top_k', 5),
        'system_prompt': data.get('system_prompt', ''),
        'documents': data.getlist('documents')
    }
    print(settings_params)

    return render_template('index.html', settings=settings_params, message="Settings saved successfully!", documents=settings_params['documents'])

@main.route('/send_message', methods=['POST'])
def send_message():
    
    langchain = current_app.config["LANGCHAIN"]
    global settings_params
    message = request.form.get('message')
    documents=settings_params['documents']
    vectorstore = langchain.connect_vectorstores(documents, settings_params['top_k'])
    response = vectorstore.similarity_search(message)
    return render_template('index.html', settings=settings_params, message=response, documents=documents)


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
    langchain.call_pgvector("org_1", "project_1", files, chunk_size, chunk_overlap)



    
    return redirect(url_for('main.index'))

@main.route('/add/<username>')
def add_user(username):
    db.session.add(User(username=username))
    db.session.commit()
    return redirect(url_for("main.index"))

