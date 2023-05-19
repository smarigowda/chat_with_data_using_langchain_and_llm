import os
import glob
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.document_loaders import PDFMinerLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from constants import CHROMA_SETTINGS

loader_mappings = {
    ".pdf": (PDFMinerLoader, {}),
    ".csv": (CSVLoader, {}),
    ".txt": (TextLoader, {}),
}

load_dotenv()


# load a single document from file_path
def load_sigle_document(file_path: str) -> Document:
    file_extension = "." + file_path.rsplit(".", 1)[-1]
    if file_extension in loader_mappings:
        loader_class, loader_args = loader_mappings[file_extension]
        loader = loader_class(file_path, **loader_args)
        return loader.load()[0]
    raise Exception(f"Unsupported file extension {file_extension}")


# load all documents from source_dir
def load_all_documents(source_dir: str) -> list[Document]:
    all_files = []
    for ext in loader_mappings:
        all_files.extend(
            glob.glob(os.path.join(source_dir, f"**/*{ext}"), recursive=True)
        )
    return [load_sigle_document(file_path) for file_path in all_files]


# main function
def main():
    # load documents and split into chunks
    persist_dir = os.getenv("PERSIST_DIR")
    source_dir = os.getenv("SOURCE_DIR")
    embeddings_model_name = os.getenv("EMBEDDINGS_MODEL_NAME")
    chunk_size = 500
    chunk_overlap = 50
    documents = load_all_documents(source_dir)
    print(f"Loaded {len(documents)} documents form {source_dir} directory")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    texts = text_splitter.split_documents(documents)
    print(f"Split into {len(texts)} chunks of text. Max chunk size is {chunk_size}")
    # create embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    # create and store vectorstore
    db = Chroma.from_documents(
        texts,
        embeddings,
        persist_directory=persist_dir,
        client_settings=CHROMA_SETTINGS,
    )
    db.persist()
    db = None
    pass


if __name__ == "__main__":
    main()
