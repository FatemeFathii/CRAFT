# Al_enable-course-recommendation-system

This repository releases the code of my Thesis work **AI-Enabled Personalized Course Recommendation System for Career and Skills Development**

## Dataset Preparation

### 1. Course Dataset

```
python ./Dataset/Course_dataset/dataset.py --dataset {'AW' or 'Udemy'} --filepath {path to save the dataset}
```
The final output is `Udemy_courses.csv` or `AW_courses.csv` in the filepath

### 2. Groud Truth Occupation Course Dataset

```
python ./Dataset/Groud_truth_occupation_course_dataset/dataset.py --filepath {path to save the dataset}
```
The final output is `occupation_course_info.csv` in the filepath

### 3. Skill Entities Annotation Dataset

First sample the course descriptions from Course dataset, and generate jsonl files for NER annotation

```
python ./Dataset/Skill_entities_annotation_dataset/preprocess.py --paths aw_courses.csv udemy_courses.csv --fractions  0.8 0.2 --num_files 8 filepath {path to save the jsonl files}
```

- `--paths`: paths of course information from Course dataset for sampling the course descriptions for NER annotation
- `--fractions`: sample fraction of each file
- `--num_files`: number of output jsonl files

Use Prodigy tool to annotate NER dataset. More information about Prodigy can be accessed from https://prodi.gy/
```
prodigy ner.manual ner_skill de_dep_news_trf ./course_descriptions.jsonl --label SKILL
```

### 4. Synthetic Recommendation Explanation Dataset

## Model Training

### BERT Domain Pretraining

### BERT-base Skill Extractor

### BERT-base Course Retriever

### LLM Ranker

### LLM Explanation Generation

## System implement