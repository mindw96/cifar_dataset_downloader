import numpy as np
from PIL import Image
import os
import pickle


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


with open('batches.meta', 'rb') as infile:
    data = pickle.load(infile, encoding='latin1')
    classes = data['label_names']

# make folder
os.mkdir('./train')
os.mkdir('./test')
for name in classes:
    os.mkdir('./train/{}'.format(name))
    os.mkdir('./test/{}'.format(name))

# Trainset Unpacking
for i in range(1, 6):
    print('Unpacking Train File {}/{}'.format(i, 5))
    train_file = unpickle('data_batch_{}'.format(i))

    train_data = train_file[b'data']

    train_data_reshape = np.vstack(train_data).reshape((-1, 3, 32, 32))
    train_data_reshape = train_data_reshape.swapaxes(1, 3)
    train_data_reshape = train_data_reshape.swapaxes(1, 2)

    train_labels = train_file[b'labels']

    train_filename = train_file[b'filenames']

    for idx in range(10000):
        train_label = train_labels[idx]
        train_image = Image.fromarray(train_data_reshape[idx])

        train_image.save('./train/{}/{}'.format(classes[train_label], train_filename[idx].decode('utf8')))

# -----------------------------------------------------------------------------------------

# Testset Unpacking
print('Unpacking Test File')
test_file = unpickle('test_batch')

test_data = test_file[b'data']

test_data_reshape = np.vstack(test_data).reshape((-1, 3, 32, 32))
test_data_reshape = test_data_reshape.swapaxes(1, 3)
test_data_reshape = test_data_reshape.swapaxes(1, 2)

test_labels = test_file[b'labels']

test_filename = test_file[b'filenames']

for idx in range(10000):
    test_label = test_labels[idx]
    test_image = Image.fromarray(test_data_reshape[idx])
    test_image.save('./test/{}/{}'.format(classes[test_label], test_filename[idx].decode('utf8')))

print('Unpacking Finish')
