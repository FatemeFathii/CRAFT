import os

# Redis configuration
#REDIS_HOST = os.getenv('REDIS_HOST', 'redis')
REDIS_HOST = os.getenv('REDIS_HOST', '0.0.0.0')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))  
REDIS_DB = int(os.getenv('REDIS_DB', 0))

# Model and embedding configuration
#MODEL_NAME = os.getenv('MODEL_NAME', "intfloat/multilingual-e5-large-instruct")
MODEL_NAME = os.getenv('MODEL_NAME', "wt3639/EduGBERT_CourseRec")
ENCODE_KWARGS = {
    'normalize_embeddings': os.getenv('NORMALIZE_EMBEDDINGS', 'True') == 'True',  
    'convert_to_tensor': os.getenv('CONVERT_TO_TENSOR', 'True') == 'True'
}
#QUERY_INSTRUCTION = os.getenv('QUERY_INSTRUCTION', 'Find the course that relates to the given occupation and cover the skills gap')
QUERY_INSTRUCTION = os.getenv('QUERY_INSTRUCTION', '')
# Other configurations
TOP_K = int(os.getenv('TOP_K', 10))
#PERSIST_DIRECTORY = os.getenv('PERSIST_DIRECTORY', "/app/data/course_emb_db")
PERSIST_DIRECTORY = os.getenv('PERSIST_DIRECTORY', "/app/data/EduGBERT_cos_escoai")
CSV_FILE_PATH = os.getenv('CSV_FILE_PATH', '/app/data/occupations_de.csv')

REC_LORA_MODEL = os.getenv('REC_LORA_MODEL', 'wt3639/Llama-3-8B-Instruct_CourseRec_lora')
EXP_LORA_MODEL = os.getenv('EXP_LORA_MODEL', 'wt3639/Lllama-3-8B-instruct-exp-adapter')
LLM_MODEL = os.getenv('LLM_MODEL', 'meta-llama/Meta-Llama-3-8B-Instruct')
HF_TOKEN  = os.getenv("HF_TOKEN",'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')