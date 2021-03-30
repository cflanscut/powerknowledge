import shutil
import os
import sys
sys.path.append('data/')
from read_PLAID_data import read_index


def copy_file(file, old_path, new_path):
    src = os.path.join(old_path, file)
    dst = os.path.join(new_path, file)
    shutil.copy(src, dst)


def move_file(file, old_path, new_path):
    src = os.path.join(old_path, file)
    dst = os.path.join(new_path, file)
    shutil.move(src, dst)


type_index = read_index('type')
path = 'model/knowledge_model_temp/jpg'

for key in type_index:
    each_list = type_index[key]
    for i, index in enumerate(each_list):
        file_name = str(index) + '.jpg'
        jpg_list = os.listdir(path + '/total')
        if file_name not in jpg_list:
            continue
        copy_file(file_name, path + '/total', path + '/' + key)

print('finished spliting!')
