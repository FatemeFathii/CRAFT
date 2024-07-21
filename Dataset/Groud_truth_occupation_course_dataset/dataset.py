import fire
from berufenet import get_berufenet
from collector import get_aw_courses
from occupation_course_matching import get_ab_courses
from preprocess import process_aw
import os
import pandas as pd
def ensure_directory_exists(directory):
    """Ensure directory exists. If not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_data(filepath='./Dataset/Groud_truth_occupation_course_dataset/data'):
    """
    Process data given a dataset name and filepath.
    
    Args:
    dataset (str): Name of the dataset.
    filepath (str): Path to the dataset file.
    """
    get_berufenet(filepath)
    get_aw_courses(filepath)
    process_aw(filepath)
    get_ab_courses(filepath)
    data1 = pd.read_csv(os.path.join(filepath,'occupation_course_info_aw.csv'))
    data2 = pd.read_csv(os.path.join(filepath,'occupation_course_info_ab.csv'))
    
    # 合并数据
    merged_data = pd.concat([data1, data2], ignore_index=True)
    
    # 将合并后的数据保存到新的CSV文件
    merged_data.to_csv(os.path.join(filepath,'occupation_course_info.csv'), index=False)


    

if __name__ == '__main__':
    fire.Fire(process_data)
