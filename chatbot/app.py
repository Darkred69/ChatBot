import os
import chainlit as cl
from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
import openai
from llama_index.agent.openai import OpenAIAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.readers.file import UnstructuredReader
from pathlib import Path
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import SubQuestionQueryEngine
from collections import deque
from literalai import LiteralClient
import nest_asyncio
from pathlib import Path
from uuid import uuid4
from collections import deque
import datetime
from typing import Optional
from llama_index.agent.openai import OpenAIAgent
import asyncio
import functools
import logging
import base64

logger = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Set API key from environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")

os.environ["LITERAL_API_KEY"] = os.getenv("LITERAL_API_KEY")
lai = LiteralClient(api_key=os.environ.get("LITERAL_API_KEY"))
lai.instrument_openai()
nest_asyncio.apply()

# Initialize reader and document storage
loader = UnstructuredReader()
doc_set = {}
all_docs = []

# Define a directory where HTML files are located
data_directory = "./data/"

async def generate_image(prompt1: str, size: str = "1024x1024") -> str:
    try:
        logger.info(f"Generating image with prompt: {prompt1}")

        response = client.images.generate(
            model="dall-e-2",
            prompt=prompt1,
            size=size,
            quality="standard",
            n=1,
        )

        image_url = response['data'][0]['url']
        logger.info("Image generated successfully")
        return image_url
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}", exc_info=True)
        return f"Error generating image: {str(e)}"

def is_image_request(prompt: str) -> bool:
    # Define keywords associated with image generation
    image_keywords = ["generate", "create", "draw", "image", "illustrate", "visualize"]
    return any(keyword in prompt.lower() for keyword in image_keywords)


async def handle_image_request(message: cl.Message):
    query = message.content
    if is_image_request(query):  # Check if it's an image request
        print("Image request detected!")
        image_url = await generate_image(query)
        response_msg = f"Here is your generated image: {image_url}"
    else:
        print("Not an image request.")
        response_msg = "This is a text-based query."

    await message.reply(response_msg)

# Check if the directory exists and list files
html_files = list(Path(data_directory).glob("*.html"))
if not html_files:
    print("No HTML files found in the directory:", data_directory)
else:
    print("Found HTML files:", [file.stem for file in html_files])

# Load all HTML files in the specified directory
for html_file in html_files:
    file_name = html_file.stem
    try:
        file_docs = loader.load_data(
            file=html_file, split_documents=False
        )
        # if not file_docs:
        #     print(f"No documents found in {html_file}")
        # else:
        #     print(f"Loaded {len(file_docs)} documents from {html_file}")
        
        for doc in file_docs:
            doc.metadata = {"file_name": file_name}
        doc_set[file_name] = file_docs
        all_docs.extend(file_docs)
    except Exception as e:
        print(f"Error loading {html_file}: {e}")

# Check if documents were loaded
# print("Loaded documents:", doc_set)

# Settings for chunk size and storage context
Settings.chunk_size = 512
index_set = {}

# Create message history for chat session
message_history = deque(maxlen=10)  # Keep the last 10 messages

# Index each document individually and store it
for file_name, documents in doc_set.items():
    if not documents:
        # print(f"No documents to index for {file_name}")
        continue

    storage_context = StorageContext.from_defaults()
    cur_index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )
    index_set[file_name] = cur_index
    Path(f"./storage/{file_name}").mkdir(parents=True, exist_ok=True)
    storage_context.persist(persist_dir=f"./storage/{file_name}")

# Reload the indexes from storage
index_set = {}
for file_name in doc_set.keys():
    storage_context = StorageContext.from_defaults(
        persist_dir=f"./storage/{file_name}"
    )
    try:
        cur_index = load_index_from_storage(
            storage_context,
        )
        index_set[file_name] = cur_index
    except Exception as e:
        print(f"Error loading index for {file_name}: {e}")

# # Check if indexes were created
# print("Loaded indexes:", index_set)

# Ensure there's at least one index available
if not index_set:
    raise ValueError("No indexes were created. Please check document loading and indexing.")

# Create individual query engine tools for each document
individual_query_engine_tools = [
    QueryEngineTool(
        query_engine=index_set[file_name].as_query_engine(),
        metadata=ToolMetadata(
            name=f"vector_index_{file_name}",
            description=f"Useful for answering queries about {file_name}",
        ),
    )
    for file_name in doc_set.keys()
]

# Add a generic fallback tool
fallback_tool = QueryEngineTool(
    query_engine=index_set[list(index_set.keys())[0]].as_query_engine(),
    metadata=ToolMetadata(
        name="fallback_tool",
        description="Fallback tool for unmatched sub-questions."
    )
)

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=individual_query_engine_tools + [fallback_tool],
    llm=OpenAI(),  # Remove the model parameter
)

async def set_sources(response, msg):
    elements = []
    label_list = []
    for count, sr in enumerate(response.source_nodes, start=1):
        elements.append(cl.Text(
            name="S" + str(count),
            content=f"{sr.node.text}",
            display="side",
            size="small",
        ))
        label_list.append("S" + str(count))
    msg.elements = elements
    await msg.update()


# Chatbot functions
@cl.on_chat_start
async def start():
    query_engine_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="sub_question_query_engine",
            description="Useful for answering queries across multiple HTML documents.",
        ),
    )

    tools = individual_query_engine_tools + [query_engine_tool]
    agent = OpenAIAgent.from_tools(tools, verbose=True)

    cl.user_session.set("query_engine", query_engine)
    await cl.Message(
        author="Assistant",
        content="Hello! I'm an AI assistant. How may I help you with your documents?"
    ).send()

@cl.on_message
async def main(message: cl.Message):
    
    # Initialize message history as an empty list if not set
    message_history = cl.user_session.get("message_history", deque(maxlen=10))
    
    if not isinstance(message_history, list):
        message_history = []
        cl.user_session.set("message_history", message_history)
    
    chat_id = str(uuid4())
    payload = {
        "chat_id": chat_id,
        "message": message.content,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }
    
    try:
        lai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[payload],
        )
        print("Message saved to Literal AI.")
    except Exception as e:
        print(f"Error saving message to Literal AI: {e}")
    
    msg = cl.Message(content="", author="Assistant")
    user_message = message.content
    
    # Process the user's query asynchronously
    res = query_engine.query(message.content)
    
    # Check if 'res' has a 'response' attribute for the full response
    if hasattr(res, 'response'):
        # Send the full response in one go
        msg.content = res.response
        message_history.append({"author": "Human", "content": user_message})
        message_history.append({"author": "AI", "content": msg.content})
        message_history = list(message_history)[-4:]  # Keep the last 4 messages
        cl.user_session.set("message_history", message_history)
    else:
        # If res does not have a 'response' attribute, output a generic message
        msg.content = "I couldn't process your query. Please try again."
    
    await msg.send()
    if res.source_nodes:
        await set_sources(res, msg)

@cl.on_chat_resume
async def resume():
    try:
        chat_history = lai.chats.list()
        if chat_history:
            for message in chat_history:
                # Append the message to local memory and send it to the chat interface
                await cl.Message(content=message['message']).send()
            print("Chat history loaded successfully!")
        else:
            print("No chat history found.")

    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        await cl.Message(content="Sorry, I was unable to load the chat history.").send()

    # Send a message indicating the session has resumed
    await cl.Message(content="Welcome back! How can I assist you today?").send()

@cl.password_auth_callback
def auth_callback(email: str, password: str) -> Optional[cl.User]:
    """Allow any email and password combination to login."""
    # Here, we simply return the user without checking any specific credentials.
    # You can add logging, validation, or other features if needed.
    if (email, password) == ("admin", "admin"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None
    
