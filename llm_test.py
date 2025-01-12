from llama_cpp import Llama    # pip install llama-cpp-python

model_path="./models/minicpm3-4b-q4_k_m.gguf"
llm = Llama(model_path=model_path)

def chat(prompt: str):
    response = llm(
        "Q: " + prompt + " Ans: ", # Prompt
        max_tokens=512, # Generate up to 32 tokens, set to None to generate up to the end of the context window
        stop=["Q:", "question:", "Question:", "</s>"], # Stop generating just before the model would generate a new question
        echo=True # Echo the prompt back in the output
    )
    return response["choices"][0]["text"]

print(chat("What is the name of the panets of solar system?"))

Q = input("Input your query: \n")
while Q != "exit":
    response = chat(Q)
    print("\n"*2)
    print("*"*30)
    print(response)
    print("*"*30)
    print("\n"*2)
    Q = input("Input your query: \n")