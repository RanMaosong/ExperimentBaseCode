import os
import glob

from tqdm import tqdm

def get_class_or_function_name(obj):
        try:
            name = obj.__name__
        except:
            name = obj.__class__.__name__
        
        return name

# def delete_blank_checkpoint(root):
#     if not os.path.isdir(root):
#         return
    
#     children = os.listdir(root)
#     if len(children) == 0:
#         os.rmdir(root)
#         return
#     for child in children:
#         path = os.path.join(root, child)
#         delete_blank_checkpoint(path)
#     if len(os.listdir(root)) == 0:
#         os.rmdir(root)
#         return


def wrap_tqdm_write(msg):
    tqdm.write(msg, end="")

def remove_file(file_name):
    drop_file = glob.glob(file_name)
    if drop_file:
        for i in drop_file:
            os.remove(i)
