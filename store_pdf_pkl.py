from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import pickle, re

# 定义函数：过滤无用的内容
def filter_text(text):
    # 保留中文字符、英文字符和常用符号
    filtered_text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9\s.,!?;×~@^&%:()\[\]{}\"\'\-—]", "", text)
    return filtered_text.strip()

def pdf2pkl(pdf_path):
    loader = PyPDFLoader(pdf_path)

    text_splitter = CharacterTextSplitter(separator = "\n\n", chunk_size = 600, chunk_overlap = 100, length_function = len)
    documents = loader.load_and_split(text_splitter)

    # # 过滤无用的内容
    for doc in documents:
        doc.page_content = filter_text(doc.page_content)

    embeddings = HuggingFaceEmbeddings(model_name="GanymedeNil/text2vec-large-chinese")
    vector_store = FAISS.from_documents(documents, embeddings) # init

    pkl_path = pdf_path[:-4] + ".pkl"

    with open(pkl_path, "wb") as f:
        pickle.dump(vector_store, f) # store