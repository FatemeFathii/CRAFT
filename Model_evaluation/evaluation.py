# get array of courses from courses.csv with columns name,description,url
from langchain_community.vectorstores import Chroma
import functools
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import csv
from io import StringIO
import os
from openai import OpenAI
from langchain.llms import HuggingFaceTextGenInference
#from langchain_mistralai.chat_models import ChatMistralAI
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import ConfigurableField
from operator import itemgetter
from langchain.prompts import HumanMessagePromptTemplate
from langchain.docstore.document import Document
import pandas as pd
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, pipeline,LlamaForCausalLM, LlamaTokenizer,BitsAndBytesConfig
from peft import PeftModel
import os
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from langchain_community.vectorstores import Chroma
from langchain import HuggingFacePipeline
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
import random
import numpy as np
import pandas as pd
import fire




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



def calculate_dcg(scores):

    return np.sum([
        (2**score - 1) / np.log2(idx + 2) for idx, score in enumerate(scores)
    ])

def calculate_ndcg_at_k(predicted_scores, k):
  
    dcg_max = calculate_dcg(sorted(predicted_scores, reverse=True)[:k])
    dcg = calculate_dcg(predicted_scores[:k])
    return dcg / dcg_max if dcg_max > 0 else 0

def calculate_mrr(predicted_scores):

    try:
        return 1 / (predicted_scores.index(1) + 1)
    except ValueError:
        return 0

def evaluate_predictions(df_actual, df_predicted):
  
    merged_df = pd.merge(df_predicted, df_actual, on=['kursnummer', 'esco_url'], how='left', indicator=True)
    merged_df['is_correct'] = merged_df['_merge'] == 'both'

    metrics = {'HR@1':[],'HR@3':[], 'HR@5': [], 'HR@10': [],'nDCG@1':[], 'nDCG@3':[], 'nDCG@5': [], 'nDCG@10': [],'MAP@1':[], 'MAP@3':[],'MAP@5':[],'MAP@10':[],'RECALL@1':[],'RECALL@3':[],'RECALL@5':[],'RECALL@10':[], 'MRR': []}


    for esco_url, group in merged_df.groupby('esco_url'):
        actual_count = df_actual[df_actual['esco_url'] == esco_url].shape[0]  
        scores = group['is_correct'].astype(int).tolist()
      
        metrics['nDCG@1'].append(calculate_ndcg_at_k(scores, 1))
        metrics['nDCG@3'].append(calculate_ndcg_at_k(scores, 3))
        metrics['nDCG@5'].append(calculate_ndcg_at_k(scores, 5))
        metrics['nDCG@10'].append(calculate_ndcg_at_k(scores, 10))
        
        
        metrics['HR@1'].append(any(scores[:1]))
        metrics['HR@3'].append(any(scores[:3]))
        metrics['HR@5'].append(any(scores[:5]))
        metrics['HR@10'].append(any(scores[:10]))
        
      
        metrics['MAP@1'].append(np.mean(scores[:1]))
        metrics['MAP@3'].append(np.mean(scores[:3]))
        metrics['MAP@5'].append(np.mean(scores[:5]))
        metrics['MAP@10'].append(np.mean(scores[:10]))

     
        correct_predictions_1 = sum(scores[:1])
        correct_predictions_3 = sum(scores[:3])
        correct_predictions_5 = sum(scores[:5])
        correct_predictions_10 = sum(scores[:10])
        metrics['RECALL@1'].append(correct_predictions_1 / actual_count if actual_count > 0 else 0)
        metrics['RECALL@3'].append(correct_predictions_3 / actual_count if actual_count > 0 else 0)
        metrics['RECALL@5'].append(correct_predictions_5 / actual_count if actual_count > 0 else 0)
        metrics['RECALL@10'].append(correct_predictions_10 / actual_count if actual_count > 0 else 0)
        metrics['MRR'].append(calculate_mrr(scores))

    return {k: np.mean(v) for k, v in metrics.items()}

def generate_result(
        prompt=None,
        tokenizer=None,
        model=None,
        device='cpu',
        first_id=0,
        second_id=1,
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
            generation_output = model.generate(
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



def compare_docs_with_context(doc_a, doc_b, row, tokenizer, model, device, first_id, second_id):
   
    courses = f"First: name: {doc_a.metadata['course_name']}  learning outcomes:{doc_a.metadata['course_skills']} Second: name: {doc_b.metadata['course_name']}  learning outcomes:{doc_a.metadata['course_skills']}" 

    target_occupation = f"name: {row['occupation_name']} description: {row['occupation_dsp']}"
    skill_gap = row['occupation_skills']
    prompt = generate_prompt(target_occupation, skill_gap, courses)
    prompt = [prompt]
    output, logit = generate_result(prompt, tokenizer, model, device, first_id, second_id)
    # Compare based on the response: [A] means doc_a > doc_b, [B] means doc_a < doc_b
    print(output, logit)
    if logit[0][0] > logit[0][1]:
        return 1  # doc_a should come before doc_b
    elif logit[0][0] < logit[0][1]:
        return -1  # doc_a should come after doc_b
    else:
        return 0  # Consider them equal if the response is unclear
    
def evaluation(course_data, test_data, course_retriver_model,LLM_model, lora_rec_adpater,hf_token, match_result, top_k=10 ):
    courses = []
    with open(course_data, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            courses.append(row)



    docs = []
    for course in courses:
        # Create document.
        doc = Document(
                page_content=course["course_name"] + ' ' + course['course_skills'],
                metadata={
                "id": course["course_id"],
                "name": course["course_name"],
                "description": course["course_dsp"],
                "skills": course["course_skills"],
                "url":  course["course_link"],
            },
        )
        docs.append(doc)


    emb_model = course_retriver_model
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    query_instruction = ''

    embedding_model = HuggingFaceBgeEmbeddings(
        model_name=emb_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        query_instruction = query_instruction)


    db = Chroma.from_documents(docs, embedding_model, collection_metadata = {"hnsw:space": "cosine"})
    retriever = db.as_retriever(search_kwargs={"k": top_k})

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    lora_weights = lora_rec_adpater

 


    tokenizer = AutoTokenizer.from_pretrained(LLM_model, token=hf_token)


    first_token = 'First'
    second_token = 'Second'

    first_id = tokenizer.convert_tokens_to_ids(first_token)
    second_id = tokenizer.convert_tokens_to_ids(second_token)
    
    model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token,

            )

    model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
                device_map={'': 0}
            )
        
    tokenizer.padding_side = "left"
        # unwind broken decapoda-research config
    #model.half()  # seems to fix bugs for some users.
    model.eval()

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2
    df_test = pd.read_csv(test_data)



    final_result = []
    for index, row in df_test.iterrows():
        occupation_query = row['occupation_inputs']

        docs = retriever.get_relevant_documents(occupation_query)
        
        partial_compare_docs = functools.partial(compare_docs_with_context, row=row, tokenizer=tokenizer, model=model, device=device, first_id=first_id, second_id=second_id)
        sorted_docs = sorted(docs, key=functools.cmp_to_key(partial_compare_docs), reverse=True)
        # Print sorted docs or use them further
        for doc in sorted_docs:
            final_result.append({
                    'course_id': doc.metadata['id'],  # 'course_id' corresponds to the header in the CSV
                    'occupation_id': row['occupation_id'],  # Assuming occupation_id is passed or available here
                })
        df_results = pd.DataFrame(final_result)

        df_results.to_csv(os.path.join(match_result,'match_result.csv'), index=False)
    


    df_actual = df_test[['course_id','occupation_id']]

    result = evaluate_predictions(df_actual, df_results)


    for metric, value in result.items():
        print(f"{metric}: {value}")
    print() 

#main
if __name__ == "__main__":
    fire.Fire(evaluation)