import os

from flask import Flask
from .extensions import db
from .routes import main

def create_app(debug=False):
    app = Flask(__name__)
    from app.langchain_copy import LangChain
    app.secret_key = os.getenv('SECRET_KEY')
    app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("DATABASE_URL")
    app.config['S3_BUCKET'] = os.getenv("S3_BUCKET_NAME")
    app.config['S3_KEY'] = os.getenv("AWS_ACCESS_KEY")
    app.config['S3_SECRET'] = os.getenv("AWS_ACCESS_SECRET")
    app.config['S3_LOCATION'] = os.getenv('S3_LOCATION')

    


    db.init_app(app)

    app.register_blueprint(main)
    langchain = LangChain()
    langchain.create_table()
    if debug:
        app.config['DEBUG'] = True


    app.config["LANGCHAIN"] = langchain
    return app