import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
import os
from langchain.prompts import PromptTemplate
from operator import itemgetter
from langchain_core.runnables import RunnableParallel

OPENAI_API_KEY = ""
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# llm used to create responses
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# use to split large texts into smaller texts
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    Helpful Answer:"""
rag_prompt_custom = PromptTemplate.from_template(template)
rag_chain_from_docs = (
        {
            "context": lambda input: format_docs(input["documents"]),
            "question": itemgetter("question"),
        }
        | rag_prompt_custom
        | llm
        | StrOutputParser()
    )

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def model_pipeline(url_path: str, question_sentence: str):
    #load data from url path
    loader = WebBaseLoader(
        web_paths=(url_path,),
    )
    docs = loader.load()

    # split large texts into smaller texts (chunked data)
    all_splits = text_splitter.split_documents(docs)

    # chunked data is converted into embeddings, these embeddings are stored into a vector database (chroma db)
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

    # retriever is used to retrieve with user query (use similarity search), return a ranked vector list
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    rag_chain_with_source = RunnableParallel(
        {"documents": retriever, "question": RunnablePassthrough()}
    ) | {
        "documents": lambda input: [doc.metadata for doc in input["documents"]],
        "answer": rag_chain_from_docs,
    }
    return rag_chain_with_source.invoke(question_sentence)["answer"]

if __name__ == "__main__":
    import time
    start_time = time.time()
    output = model_pipeline(url_path="https://policies.google.com/privacy?hl=en-US",
                            question_sentence="Why do you collect my information?")
    print(output)
    print("time: ", time.time() - start_time)