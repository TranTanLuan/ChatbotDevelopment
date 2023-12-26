from model import model_pipeline
from typing import Union
from fastapi import FastAPI
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/ask")
def ask(url_path: str, question_sentence: str):
    result = model_pipeline(url_path=url_path, question_sentence=question_sentence)
    return {"answer": result}