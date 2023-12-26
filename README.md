# Develop a chatbot to answer user questions about the website’s information
## Introduction
This project is to use RAG technique to augment LLM (GPT-3.5) with additional data, store data in a vector database (Chroma) to retrieve, apply langchain framework to build this RAG application (chatbot). This is deployed with FastAPI and Docker.

## Create api key to use OpenAI models
- Go to "https://platform.openai.com/" to sign up and create api key
- Them go to "RAG_Fastapi/model.py", fill in OPENAI_API_KEY at line 15

## Run
### Solution 1:
```
pip install -r requirements.txt
uvicorn main:app --reload
```
- "main:app" is a application in main.py with app function
- "--reload" is an optional, will restart automatically when detecting the change in source
- After run the above command: go to "http://127.0.0.1:8000/docs" to use the app  
![alt text](RAG_Fastapi/imgs/Using_the_app.png)
- Fill in the "url_path" you want to extract information.
- Fill in the "question_sentence" you want to ask.
- Finally, click "Execute" then get the result.
### Solution 2:
```
docker compose up --build
```
- After run the above command, do the same as the remaining steps in Solution 1