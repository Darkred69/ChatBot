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

# Set API key from environment
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Initialize reader and document storage
loader = UnstructuredReader()
doc_set = {}
all_docs = []

# Define a directory where HTML files are located
data_directory = "./data/"

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

# # Check if documents were loaded
# print("Loaded documents:", doc_set)

# Settings for chunk size and storage context
Settings.chunk_size = 512
index_set = {}

# Index each document individually and store it
for file_name, documents in doc_set.items():
    if not documents:
        print(f"No documents to index for {file_name}")
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
        print(f"Loaded index for {file_name}")
    except Exception as e:
        print(f"Error loading index for {file_name}: {e}")

# Check if indexes were created
print("Loaded indexes:", index_set)

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

# Chatbot functions
@cl.on_chat_start
async def start():
    query_engine = SubQuestionQueryEngine.from_defaults(
        query_engine_tools=individual_query_engine_tools + [fallback_tool],
        llm=OpenAI(model="gpt-4o-mini"),
    )

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
    query_engine = cl.user_session.get("query_engine")

    msg = cl.Message(content="", author="Assistant")

    try:
        res = await cl.make_async(query_engine.query)(message.content)

        if hasattr(res, 'response'):
            msg.content = res.response
        else:
            msg.content = "I couldn't process your query. Please try again."
    except KeyError as e:
        # Adding a debugging message to help track down unmatched tool names
        msg.content = f"I'm sorry, I don't have the answer for this question"
    except Exception as e:
        msg.content = f"An unexpected error occurred: {e}"

    await msg.send()
