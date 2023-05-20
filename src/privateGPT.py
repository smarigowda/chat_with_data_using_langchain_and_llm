import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import GPT4All

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_dir = os.environ.get("PERSIST_DIR")

model_type = os.environ.get("MODEL_TYPE")
model_path = os.environ.get("MODEL_PATH")
model_n_ctx = os.environ.get("MODEL_N_CTX")

from constants import CHROMA_SETTINGS


def main():
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS,
    )
    retriever = db.as_retriever()
    callbacks = [StreamingStdOutCallbackHandler()]
    llm = GPT4All(
        model=model_path,
        n_ctx=model_n_ctx,
        backend="gptj",
        callbacks=callbacks,
        verbose=False,
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=False
    )
    while True:
        query = input("Enter query: ")
        if query == "exit":
            break
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        print("Question: ", query)
        print("Answer: ", answer)


if __name__ == "__main__":
    main()
