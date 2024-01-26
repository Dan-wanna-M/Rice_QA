import argparse
from llama_index import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
)
from llama_index.llms import LlamaCPP
# use Huggingface embeddings
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index import StorageContext, load_index_from_storage
from llama_index.prompts  import PromptTemplate
from llama_index.llms import ChatMessage, MessageRole
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.chat_engine.condense_question import CondenseQuestionChatEngine
import llama_index.response_synthesizers

parser = argparse.ArgumentParser()
parser.add_argument("--language_model_path", type=str)
parser.add_argument("--embed_model_name", type=str)
parser.add_argument("--context_window", type=int, default=3900)
parser.add_argument("--max_new_tokens", type=int, default=256)
parser.add_argument("--n_gpu_layers", type=int)

# default_prompt = PromptTemplate("""\
# Given a conversation (between Human and Assistant) and a follow up message from Human, \
# rewrite the message to be a standalone question that captures all relevant context \
# from the conversation.

# <Chat History>
# {chat_history}

# <Follow Up Message>
# {question}

# <Standalone question>
# """)

# # list of `ChatMessage` objects
# custom_chat_history = [
#     ChatMessage(
#         role=MessageRole.USER,
#         content='Hello assistant, we are having a insightful discussion about Paul Graham today.'
#     ),
#     ChatMessage(
#         role=MessageRole.ASSISTANT,
#         content='Okay, sounds good.'
#     )
# ]

args = parser.parse_args()
llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url=None,
    # optionally, you can set the path to a pre-downloaded model instead of model_url
    model_path=args.language_model_path,
    temperature=0.1,
    max_new_tokens=args.max_new_tokens,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    context_window=args.context_window,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": args.n_gpu_layers},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    verbose=True,
)
embed_model = HuggingFaceEmbedding(model_name=args.embed_model_name)


service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model
)
documents = SimpleDirectoryReader('data2').load_data()
index = VectorStoreIndex.from_documents(documents, service_context=service_context)
query_engine = index.as_query_engine()



# chat_engine = CondenseQuestionChatEngine.from_defaults(
#     query_engine=query_engine,
#     condense_question_prompt=custom_prompt,
#     chat_history=custom_chat_history,
#     service_context=service_context,
#     verbose=True
# )

chat_engine = index.as_chat_engine(
    chat_mode='react',
    verbose=True
)
while True: 
    question = input("What is your question?")
    if question == "reset":
        chat_engine.reset()
        continue
    print(chat_engine.chat(question))