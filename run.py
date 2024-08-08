import os, logging, sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

from huggingface_hub import login

# os.environ["HF_KEY"] = "Your Hugging Face access token goes here"
# login(token=os.environ.get('HF_KEY'),add_to_git_credential=True)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"

embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME, embed_batch_size=10)

print(embed_model)

from llama_index.core import Settings

Settings.embed_model = embed_model
Settings.chunk_size = 256
Settings.chunk_overlap = 50

from llama_index.core import PromptTemplate

system_prompt = """<|SYSTEM|># You are an AI-enabled admin assistant.
Your goal is to answer questions accurately using only the context provided.
"""

# This will wrap the default prompts that are internal to llama-index
query_wrapper_prompt = PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>")

LLM_MODEL_NAME = "Meta-Llama-3.1-8B-bnb-4bit"


import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,  # or load_in_4bit=True
    llm_int8_enable_fp32_cpu_offload=True,  # This enables CPU offloading
)

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=512,
    generate_kwargs={"temperature": 0.1, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name=LLM_MODEL_NAME,
    model_name=LLM_MODEL_NAME,
    device_map="cpu",  # Ensure it's using CPU
    quantization_config=quantization_config
)

Settings.llm = llm