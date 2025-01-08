from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "llama3",
    temperature = 0.8,
    num_predict = 256,
    # other params ...
)


messages = [
    ("system", "You are a helpful translator. Translate the user sentence to German."),
    ("human", "I love programming."),
]
response = [chunk.content for chunk in llm.stream(messages)]
print("".join(response))
