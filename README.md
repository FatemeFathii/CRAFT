# Al_enable-course-recommendation-system

This repository releases the code of my Thesis work **AI-Enabled Personalized Course Recommendation System for Career and Skills Development**

## Dataset Preparation

### 1. Course Dataset

```
python ./Dataset/Course_dataset/dataset.py --dataset {'AW' or 'Udemy'} --filepath {path to save the dataset}
```
The final output is `Udemy_courses.csv` or `AW_courses.csv` in the filepath


### 2. Skill Entities Annotation Dataset

First sample the course descriptions from Course dataset, and generate jsonl files for NER annotation

```
python ./Dataset/Skill_entities_annotation_dataset/preprocess.py --paths aw_courses.csv udemy_courses.csv --fractions  0.8 0.2 --num_files 8 filepath {path to save the jsonl files}
```

- `--paths`: paths of course information from Course dataset for sampling the course descriptions for NER annotation
- `--fractions`: sample fraction of each file
- `--num_files`: number of output jsonl files

Use Prodigy tool to annotate NER dataset. More information about Prodigy can be accessed from https://prodi.gy/

```
prodigy ner.manual ner_skill de_dep_news_trf ./course_description_1.jsonl --label SKILL
```

### 3. Groud Truth Occupation Course Dataset

```
python ./Dataset/Groud_truth_occupation_course_dataset/dataset.py --filepath {path to save the dataset} --model_name {skill extractor model name}
```
The final output is `occupation_course_info.csv` in the filepath


### 4. Synthetic Recommendation Explanation Dataset

```
python ./Dataset/Synthetic_recommendation_explanation _dataset/dataset.py --LLM-key {OpenAI api key} --occupation_course_data {path of Groud Truth occupation course dataset} --filepath {path to save the dataset}
```

## Model Training

### 1. BERT Domain Pretraining

```
torchrun --nproc_per_node {number of gpus} -m ./Model_training/BERT_domian_pretraining/run.py \
--output_dir {output path} \
--model_name_or_path {base model} \
--train_data {domain corpora jsonl file} \
--learning_rate 2e-5 \
--fp16 \
--num_train_epochs 5 \
--per_device_train_batch_size 16 \
--dataloader_drop_last True \
--max_seq_length 512 \
--logging_steps 10 \
--dataloader_num_workers 8 \
--save_steps 2000
```

After training, the encoder model will saved to {output_dir}/encoder_model

### 2. BERT-base Skill Extractor

```
ner_course_train,eval:ner_course_test {output path} \
--epochs 100 \
--model-name {domain pretrained model} \
--batch-size 32 \
-lr 5e-6 
```

### 3. BERT-base Course Retriever

```
torchrun --nproc_per_node {number of gpus} -m ./Model_training/BERT_base_course_retriever \
--output_dir {output path} \
--model_name_or_path {domain pretrained model} \
--train_data {training set} \
--learning_rate 1e-5 \
--fp16 \
--num_train_epochs 100 \
--per_device_train_batch_size 4 \
--dataloader_drop_last True \
--normlized True \
--temperature 0.02 \
--query_max_len 360 \\
--passage_max_len 512 \
--train_group_size 15 \
--negatives_cross_device \
--logging_steps 10 \
--save_steps 1000 \
--query_instruction_for_retrieval
```

### 4. LLM Ranker

```
CUDA_VISIBLE_DEVICES=$1 python -u ./Model_training/LLM_ranker/finetune_rec.py \
                    --base_model {base LLM model such as Meta-Llama-3-8B-Instruct} \
                    --train_data_path {training set} \
                    --val_data_path {validation set} \
                    --output_dir {output path} \
                    --batch_size 16 \
                    --micro_batch_size 8 \
                    --num_epochs 4 \
                    --learning_rate 1e-4 \
                    --cutoff_len 1500 \
                    --lora_r 16 \
                    --lora_alpha 16\
                    --lora_dropout 0.05 \
                    --lora_target_modules '[q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj]' \
                    --train_on_inputs \
                    --group_by_length 
```

### 5. LLM Explanation Generation

```
CUDA_VISIBLE_DEVICES=$1 python -u ./Model_training/LLM_explanation_generation/finetune.py \
                    --base_model {base LLM model such as Meta-Llama-3-8B-Instruct} \
                    --train_data_path {training set} \
                    --val_data_path {validation set} \
                    --output_dir {output path} \
                    --batch_size 4 \
                    --micro_batch_size 2 \
                    --num_epochs 30 \
                    --learning_rate 1e-4 \
                    --cutoff_len 2000 \
                    --lora_r 16 \
                    --lora_alpha 16\
                    --lora_dropout 0.05 \
                    --lora_target_modules '[q_proj,v_proj,k_proj,o_proj,gate_proj,up_proj,down_proj]' \
                    --train_on_inputs \
                    --group_by_length 
```

## System implement