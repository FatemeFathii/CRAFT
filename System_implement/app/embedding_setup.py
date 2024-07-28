from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

from langchain.docstore.document import Document
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig,BitsAndBytesConfig
from peft import PeftModel
from config import *
import os
import torch

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

os.environ['CURL_CA_BUNDLE'] = ""
embedding_int = HuggingFaceBgeEmbeddings(
    model_name=MODEL_NAME,
    encode_kwargs=ENCODE_KWARGS,
    query_instruction=QUERY_INSTRUCTION
)

embedding_sim = HuggingFaceBgeEmbeddings(
    model_name=MODEL_NAME,
    encode_kwargs=ENCODE_KWARGS,
    query_instruction='Retrieve semantically similar text.'
)

db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embedding_int)
retriever = db.as_retriever(search_kwargs={"k": TOP_K})



lora_weights_rec = REC_LORA_MODEL
lora_weights_exp = EXP_LORA_MODEL


tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, token=HF_TOKEN)


first_token = 'First'
second_token = 'Second'
# 获取token的ID
first_id = tokenizer.convert_tokens_to_ids(first_token)
second_id = tokenizer.convert_tokens_to_ids(second_token)
model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            token=HF_TOKEN,
        )

rec_adapter = PeftModel.from_pretrained(
            model,
            lora_weights_rec
        )
      

tokenizer.padding_side = "left"
    # unwind broken decapoda-research config
#model.half()  # seems to fix bugs for some users.
rec_adapter.eval()

rec_adapter.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
rec_adapter.config.bos_token_id = 1
rec_adapter.config.eos_token_id = 2



def generate_prompt(target_occupation, skill_gap, courses):
    return f"""
### Instruction:
"As an education expert, you have been provided with a target occupation, a skill gap, and information on two candidate courses. Your task is to determine which course better matches the target occupation and skill gap. Please respond with 'First' or 'Second' to indicate your recommendation.

### Input:
Target Occupation: {target_occupation}
Skill Gap: {skill_gap}
candidate courses: {courses}

### Response:
"""
'''
prompt_re = ChatPromptTemplate.from_template(template_re)
chain_re = (
    runnable
    | prompt_re
)
'''
def evaluate(
        prompt=None,
        temperature=0,
        top_p=1.0,
        top_k=40,
        num_beams=1,
        max_new_tokens=120,
        batch_size=1,
        **kwargs,
    ):

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            rec_adapter.to(device)
            generation_output = rec_adapter.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                # batch_size=batch_size,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        scores = generation_output.scores[0].softmax(dim=-1)
        logits = torch.tensor(scores[:,[first_id, second_id]], dtype=torch.float32).softmax(dim=-1)
        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split('Response:\n')[-1] for _ in output]
        return output, logits.tolist()

def compare_docs_with_context(doc_a, doc_b, target_occupation_name, target_occupation_dsp,skill_gap):
    
    #courses = f"First: name: {doc_a.metadata['name']}  description:{doc_a.metadata['description']} Second: name: {doc_b.metadata['name']}  description:{Sdoc_b.metadata['description']}" 
    courses = f"First: name: {doc_a.metadata['name']}  learning outcomes:{doc_a.metadata['skills'][:1500]} Second: name: {doc_b.metadata['name']}  learning outcomes:{doc_b.metadata['skills'][:1500]}" 
    target_occupation = f"name: {target_occupation_name} description: {target_occupation_dsp[:1500]}"
    skill_gap = skill_gap
    prompt = generate_prompt(target_occupation, skill_gap, courses)
    prompt = [prompt]
    output, logit = evaluate(prompt)
    # Compare based on the response: [A] means doc_a > doc_b, [B] means doc_a < doc_b
    print(output, logit)
    if logit[0][0] > logit[0][1]:
        return 1  # doc_a should come before doc_b
    elif logit[0][0] < logit[0][1]:
        return -1  # doc_a should come after doc_b
    else:
        return 0  # Consider them equal if the response is unclear


#-----------------------------------------explanation-------------------------------------
exp_adapter = PeftModel.from_pretrained(
            model,
            lora_weights_exp
        )
exp_adapter.eval()

exp_adapter.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
exp_adapter.config.bos_token_id = 1
exp_adapter.config.eos_token_id = 2

def generate_prompt_exp(input_text):
    return f"""
### Instruction:
As an education expert, you have been provided with information on target occupations and skills gaps, along with recommended course details. Your task is to explain the recommendation in German, focusing on how the course's learning outcomes and target skills relate to the identified skills gaps.

### Input:
{input_text}

### Response:
"""

def generate_exp(
        prompt=None,
        temperature=0,
        top_p=1.0,
        top_k=40,
        num_beams=1,
        max_new_tokens=512,
        batch_size=1,
        **kwargs,
    ):

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            exp_adapter.to(device)
            generation_output = exp_adapter.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                # batch_size=batch_size,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )
        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split('Response:\n')[-1] for _ in output]
        return output


def find_similar_occupation(target_occupation_query, berufe, top_k, similarity_func):
  
    # Pro kurs wird ein Document erstellt. Dieses enthält Metadaten sowie einen page_content. 
    # Der Inhalt von page_content wird embedded und so für die sucher verwendet.
    docs = []
    for index, beruf in berufe.iterrows():  
        # Create document.
        doc = Document(
            page_content= beruf['short name'] + ' ' + beruf['full name'] + ' ' + beruf['description'],  
            metadata={
                "id": beruf["id"],
                "name": beruf['short name'],
                "description": beruf["description"],
                "entry_requirements": beruf["entry requirements"]
            },
        )
        docs.append(doc)
    
    db_temp = Chroma.from_documents(documents = docs, embedding= embedding_sim, collection_metadata = {"hnsw:space": similarity_func})
    # Retriever will search for the top_5 most similar documents to the query.
    retriever_temp = db_temp.as_retriever(search_kwargs={"k": top_k})
    top_similar_occupations = retriever_temp.get_relevant_documents(target_occupation_query)

    return top_similar_occupations