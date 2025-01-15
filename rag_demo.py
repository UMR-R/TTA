import pickle
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA, ConversationalRetrievalChain

with open("./_resources/docs/SoftEngineer.pkl", "rb") as f:
    vector_store = pickle.load(f) # load

# Prompt
def Prompt(query):
    prompt = "你是一个问答机器人，你需要回答我的问题，请注意不要提出新的问题。\n问题是：" + query
    return prompt

model_path="./models/minicpm3-4b-q4_k_m.gguf"
llm = LlamaCpp(model_path=model_path, 
               n_ctx=2048, # n_ctx: 输入的token最大长度
               temperature=0.7, 
               stop=["Q:", "question:", "Question:", "Unhelpful Answer:", "Unfriendly Answer:", "</s>"], # 生成停止标志
               n_gpu_layers=1, # 部署在GPU上的层数，需要在安装llama-cpp-python时指定编译参数支持GPU
               ) 
retriever = vector_store.as_retriever()

# QA式检索
def get_basic_qa_chain():
    chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff",
        retriever=retriever, 
        verbose=True,
    )
    return chain

# 对话式检索
def get_conversational_retrieval_chain():
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True, 
        input_key="question", 
        output_key="source_documents"
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm, 
        retriever=retriever, 
        memory=memory, 
        return_source_documents=True, 
    )
    return chain

chain = get_basic_qa_chain()
query = "成绩组成是什么？"
while query != "exit":
    query = Prompt(query)
    result = chain.invoke(query)
    print("*"*20)
    res = result['result'].split("output:")[-1]
    print("Res: \n" + res)
    print("*"*20)
    query = input("input your query:\n")