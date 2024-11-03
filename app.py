import chainlit as cl
from src.ibusiness_agentRAG import app


welcome_message = "¡Welcome! ¿How can I help you?"
@cl.on_chat_start
async def start_chat():

    print("Initialised chain...")

    await cl.Message(content=welcome_message).send()
    cl.user_session.set("runnable", app)


@cl.on_message
async def main(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    msg = cl.Message(content="")

    input = {"question": message.content}

    for output in runnable.stream(input):
        for key, value in output.items():
            print(f"Finished running: {key}:")
            if key == "generate":
                answer = value["answer"]
                await msg.stream_token(answer)

    await msg.send()
#what you know about artificial intelligence?
#what you know about albums?
#what you know about albums and artificial intelligence?