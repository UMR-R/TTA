import os
import pickle
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA

from store_pdf_pkl import pdf2pkl

class RAG:
    """
    Using LLM and local files to build a RAG system.

    You can choose the llm models from https://huggingface.co/models?library=gguf&sort=trending or other models zoo.
    To build this system on your personal PC, the gguf model less than 14B is recommended.

    Args:
        files_path: Root_path of your files.
        models_path[optional]: Your local gguf models' dir path if you use local model file else None(default).
        intoken_length[optional]: Max window size of your input token. Default: 2048.
        gpu[optional]: How many layers will be put on your GPU. To use this param, you need install GPU version llama-cpp-python. Default: 0.
        history[optional]: LLM's memory function. 
    """
    
    stop=["Q:", "question:", "Question:", "Unhelpful Answer:", "Unfriendly Answer:", "</s>"]
    
    def __init__(self, files_path, models_path=None, intoken_length=2048, gpu=0, history=False):
        self.files_path = files_path
        self.models_path = models_path
        self.intoken_length = intoken_length
        self.gpu = gpu
        self.history = history

    def register_llm(self, model):
        """
        Args:
            model: The model(huggingface) or model_file_name(local) you use.
        """

        if self.models_path:
            model = os.path.join(self.models_path, model)
        self.llm = LlamaCpp(model_path=model, 
                            n_ctx=self.intoken_length, # n_ctx: 输入的token最大长度
                            temperature=0.7, 
                            stop=self.stop, # 生成停止标志
                            n_gpu_layers=self.gpu, # 部署在GPU上的层数，需要在安装llama-cpp-python时指定编译参数支持GPU
                            )

    def upload_file(self, file):
        """
        Args:
            file: PDF file name. If you have pkl file, you don't need to call function upload_file, just function use_file.
        """
        try:
            pdf2pkl(os.path.join(self.files_path, file))
        except ValueError:
            print("\033[91mPDF path error!\033[0m")
    
    def use_file(self, file):
        """
        Args:
            file: PKL file name.
        """
        with open(os.path.join(self.files_path, file), 'rb') as f:
            vector_store = pickle.load(f)
        retriever = vector_store.as_retriever()
        self.chain = RetrievalQA.from_chain_type(
            llm=self.llm, 
            chain_type="stuff",
            retriever=retriever, 
            verbose=True,
        )

    def Prompt(self, query):
        prompt = "你是一个问答机器人，你需要回答我的问题，请注意不要提出新的问题。\n问题是：" + query
        return prompt

    def __call__(self, query):
        """
        Args:
            Your query to LLM.
        """
        Q = self.Prompt(query)
        result = self.chain.invoke(Q)
        print("*"*20)
        print("Q: " + query + "\n")
        print("A: " + result['result'].split("output:")[-1])
        print("*"*20)

if __name__ == "__main__":
    rag = RAG("./_resources/docs", "./models")
    rag.register_llm("minicpm3-4b-q4_k_m.gguf")
    rag.use_file("SoftEngineer.pkl")
    rag("成绩组成是什么？")