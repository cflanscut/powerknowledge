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
train_size = 0.8
test_size = 0.3
old_path = 'data/source/submetered_zengj/total'
# new_path = 'data/source/submetered_new_pured/feature_process/total'
test_path = 'data/source/submetered_zengj/testing'
train_path = 'data/source/submetered_zengj/training'
validation_path = 'data/source/submetered_zengj/validation'
for key in type_index:
    each_list = type_index[key]
    each_len = len(each_list)
    train_len2 = int((1 - test_size) * each_len)
    train_len1 = int((train_size) * train_len2)
    csv_list = os.listdir(old_path)
    for i in range(train_len1):
        file_name = str(each_list[i]) + '.csv'
        if file_name not in csv_list:
            continue
        copy_file(file_name, old_path, train_path)
        # move_file(file_name, old_path, new_path)
    for i in range(train_len1, train_len2):
        file_name = str(each_list[i]) + '.csv'
        if file_name not in csv_list:
            continue
        copy_file(file_name, old_path, validation_path)
        # move_file(file_name, old_path, new_path)
    for i in range(train_len2, each_len):
        file_name = str(each_list[i]) + '.csv'
        if file_name not in csv_list:
            continue
        copy_file(file_name, old_path, test_path)
        # move_file(file_name, old_path, new_path)

print('training data:%03d' % (len(os.listdir(train_path))))
print('validation data:%03d' % (len(os.listdir(validation_path))))
print('testing data:%03d' % (len(os.listdir(test_path))))
print('finished spliting!')
