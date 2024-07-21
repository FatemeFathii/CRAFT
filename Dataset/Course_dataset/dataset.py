import fire
from aw_dataset import aw_get_courses
from udemy_dataset import udemy_get_courses
import os
def ensure_directory_exists(directory):
    """Ensure directory exists. If not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_data(dataset, filepath='./Dataset/Course_dataset/data'):
    """
    Process data given a dataset name and filepath.
    
    Args:
    dataset (str): Name of the dataset.
    filepath (str): Path to the dataset file.
    """
    ensure_directory_exists(filepath)
    if dataset == 'AW':
        aw_get_courses(filepath)
    elif dataset == 'Udemy':
        udemy_get_courses(filepath)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    

if __name__ == '__main__':
    fire.Fire(process_data)
