import os

from flask import Flask
from .extensions import db
from .routes import main

def create_app():
    app = Flask(__name__)
    from app.langchain import LangChain

    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
    app.config['UPLOAD_FOLDER'] = 'uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)



    db.init_app(app)

    app.register_blueprint(main)
    langchain = LangChain()
    

    app.config["LANGCHAIN"] = langchain
    return app