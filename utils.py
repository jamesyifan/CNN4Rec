import pandas as pd
import xlrd
import numpy as np
import tensorflow as tf
import os

IMG_ROOT_OLD= 'img/'
IMG_ROOT = 'img_new/'
LABEL_ROOT = 'label/'
TF_ROOT = 'tfrecord/'

LABEL = 'label.xlsx'
DEFAULT_IMG_HEIGHT = 256
DEFAULT_IMG_WIDTH = 100
DEFAULT_LABEL_SIZE = 83

def load_label():
    workbook = xlrd.open_workbook(LABEL_ROOT+LABEL)
    booksheet = workbook.sheet_by_name('Sheet2')
    language_dict = {}
    style_dict = {}
    song_dict = {}
    sensibility_dict = {}
    reason_dict = {}
    for row in range(booksheet.nrows):
        if row == 0:
            continue
        language_value = booksheet.cell_value(row, 2).strip('##').split('##')
        style_value = booksheet.cell_value(row, 5).strip('##').split('##')
        song_value = booksheet.cell_value(row, 6).strip('##').split('##')
        sensibility_value = booksheet.cell_value(row, 7).strip('##').split('##')
        reason_value = booksheet.cell_value(row, 8).strip('##').split('##')
        for value in language_value:
            if value != u'':
                if value not in language_dict.keys():
                    language_dict[value] = len(language_dict) 
        for value in style_value:
            if value != u'':
                if value not in style_dict.keys():
                    style_dict[value] = len(style_dict)
        for value in song_value:
            if value != u'':
                if value not in song_dict.keys():
                    song_dict[value] = len(song_dict)
        for value in sensibility_value:
            if value != u'':
                if value not in sensibility_dict.keys():
                    sensibility_dict[value] = len(sensibility_dict)
        for value in reason_value:
            if value != u'':
                if value not in reason_dict.keys():
                    reason_dict[value] = len(reason_dict)
    label = np.zeros([booksheet.nrows, len(reason_dict)+1], int)    
    for row in range(booksheet.nrows):
        if row == 0:
            continue
        reason_value = booksheet.cell_value(row, 8).strip('##').split('##')
        id_value = int(booksheet.cell_value(row, 0))
        for value in reason_value:
            if value != u'':
                label[row, reason_dict[value]] = 1
        label[row, len(reason_dict)] = id_value
    return label

#def : 


        #row_data = []
        #    cel = booksheet.cell(row, col)
        #    val = cel.value
        #    try:
        #        val
        #    except:
        #        pass
    #dict = pd.ExcelFile(ROOT+LABEL, names = ['id', 'name', 'lan_dict', 'sty_dict', 'song_dict', 'induc_dict', 'sens_dict', 'album', 'singer', 'time', 'rate']).parse()
    #for index, row in dict.iterrows():
    #    print row
    #    if index == 0:
    #        break
def get_slice_dims(input_img):
    img_width = input_img.shape[0]
    img_height = input_img.shape[1]
    num_slices = img_width // DEFAULT_IMG_WIDTH 
    unused_size = img_width - (num_slices * DEFAULT_IMG_WIDTH)
    start_px = 0 + unused_size
    image_dims = []
    for i in range(num_slices):
        img_width = DEFAULT_IMG_WIDTH
        image_dims.append((start_px, start_px + DEFAULT_IMG_WIDTH))
        start_px += DEFAULT_IMG_WIDTH
    return image_dims
def slice_spect():
    for input_file in os.listdir(IMG_ROOT_OLD):  
        input_file_cleaned = input_file.replace('.npy', '')
        img = np.load(IMG_ROOT_OLD + input_file)
        dims = get_slice_dims(img)
        counter = 0
        for dim in dims:
            counter_formatted = str(counter).zfill(3)
            img_name = '{}_{}.npy'.format(input_file_cleaned, counter_formatted)
            start_width = dim[0]
            end_width = dim[1]
            sliced_img = img[start_width:end_width, :]
            np.save(IMG_ROOT+img_name, sliced_img)
            counter += 1
def load_train(b_size, n_epochs):
    fileNameQue = tf.train.string_input_producer([TF_ROOT + 'train.tfrecords'])#, num_epochs=n_epochs)
    reader = tf.TFRecordReader()
    key, value = reader.read(fileNameQue)
    features = tf.parse_single_example(value, features={
    'label': tf.FixedLenFeature([], tf.string),
    'labelsize': tf.FixedLenFeature([], tf.int64),
    'img':tf.FixedLenFeature([], tf.string)
    })
    img = tf.decode_raw(features['img'], tf.int64)
    img = tf.reshape(img, [DEFAULT_IMG_WIDTH, DEFAULT_IMG_HEIGHT, 1])
    label = tf.decode_raw(features['label'], tf.int64)
    labelsize = features['labelsize']
    label = tf.reshape(label, [DEFAULT_LABEL_SIZE])
    labels, labelsizes, imgs = tf.train.shuffle_batch(
        [label, labelsize, img], 
        batch_size=b_size,
        capacity=2000,
        num_threads=10,
        min_after_dequeue=100)
    return imgs, labels
    #return img, label
def load_valid(b_size):
    fileNameQue = tf.train.string_input_producer([TF_ROOT + 'valid.tfrecords'])
    reader = tf.TFRecordReader()
    key, value = reader.read(fileNameQue)
    features = tf.parse_single_example(value, features={
    'label': tf.FixedLenFeature([], tf.string),
    'labelsize': tf.FixedLenFeature([], tf.int64),
    'img':tf.FixedLenFeature([], tf.string)
    })
    img = tf.decode_raw(features['img'], tf.int64)
    img = tf.reshape(img, [DEFAULT_IMG_WIDTH, DEFAULT_IMG_HEIGHT, 1])
    label = tf.decode_raw(features['label'], tf.int64)
    labelsize = features['labelsize']
    label = tf.reshape(label, [DEFAULT_LABEL_SIZE])
    labels, labelsizes, imgs = tf.train.shuffle_batch(
        [label, labelsize, img], 
        batch_size=b_size,
        capacity=2000,
        num_threads=10,
        min_after_dequeue=100)
    return imgs, labels

def load_test(b_size):
    fileNameQue = tf.train.string_input_producer([TF_ROOT + 'test.tfrecords'])
    reader = tf.TFRecordReader()
    key, value = reader.read(fileNameQue)
    features = tf.parse_single_example(value, features={
    'label': tf.FixedLenFeature([], tf.string),
    'labelsize': tf.FixedLenFeature([], tf.int64),
    'img':tf.FixedLenFeature([], tf.string)
    })
    img = tf.decode_raw(features['img'], tf.int64)
    img = tf.reshape(img, [DEFAULT_IMG_WIDTH, DEFAULT_IMG_HEIGHT, 1])
    label = tf.decode_raw(features['label'], tf.int64)
    labelsize = features['labelsize']
    label = tf.reshape(label, [DEFAULT_LABEL_SIZE])
    labels, labelsizes, imgs = tf.train.shuffle_batch(
        [label, labelsize, img], 
        batch_size=b_size,
        capacity=2000,
        num_threads=10,
        min_after_dequeue=100)
    return imgs, labels
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def split_data(train, valid, test):
    train_record = TF_ROOT + 'train.tfrecords'
    valid_record = TF_ROOT + 'valid.tfrecords'
    test_record = TF_ROOT + 'test.tfrecords'
    train_writer = tf.python_io.TFRecordWriter(train_record)
    valid_writer = tf.python_io.TFRecordWriter(valid_record)
    test_writer = tf.python_io.TFRecordWriter(test_record)
    train_num = 0
    valid_num = 0
    test_num = 0
    for img_name in os.listdir(IMG_ROOT):  
        img_name_cleaned = img_name.replace('.npy', '')
        img_name_cleaned = int(img_name_cleaned.split('_')[0])
        if img_name_cleaned in train[:,-1]:
            train_num += 1
            train_img_raw = np.load(IMG_ROOT+img_name).tostring()
            train_label_index = np.where(train[:,-1]==img_name_cleaned)
            train_label = train[train_label_index[0][0],:-1].tostring()
            train_example = tf.train.Example(features=tf.train.Features(feature={
            'label':_bytes_feature(train_label),
            'labelsize':_int64_feature(train[train_label_index[0][0],:-1].shape[0]),
            'img': _bytes_feature(train_img_raw)}))
            train_writer.write(train_example.SerializeToString())
        elif img_name_cleaned in valid[:,-1]:
            valid_num += 1
            valid_img_raw = np.load(IMG_ROOT+img_name).tostring()
            valid_label_index = np.where(valid[:,-1]==img_name_cleaned)
            valid_label = valid[valid_label_index[0][0],:-1].tostring()
            valid_example = tf.train.Example(features=tf.train.Features(feature={
            'label':_bytes_feature(valid_label), 
            'labelsize':_int64_feature(train[train_label_index[0][0],:-1].shape[0]),
            'img': _bytes_feature(valid_img_raw)}))
            valid_writer.write(valid_example.SerializeToString())
        elif img_name_cleaned in test[:,-1]: 
            test_num += 1
            test_img_raw = np.load(IMG_ROOT+img_name).tostring()
            test_label_index = np.where(test[:,-1]==img_name_cleaned)
            test_label = test[test_label_index[0][0],:-1].tostring()
            test_example = tf.train.Example(features=tf.train.Features(feature={
            'label':_bytes_feature(test_label), 
            'labelsize':_int64_feature(len(test_label)),
            'img': _bytes_feature(test_img_raw)}))
            test_writer.write(test_example.SerializeToString())
    train_writer.close()
    valid_writer.close()
    test_writer.close()
    return train_num, valid_num, test_num
        
def load_data():
    label = load_label()
    #slice_spect()
    train_num =  int(label.shape[0] * 0.8)
    valid_num = int(label.shape[0] * 0.1)
    test_num = label.shape[0]-train_num-valid_num
    np.random.shuffle(label)
    #train_num, valid_num, test_num = split_data(label[0:train_number, :], label[train_number:(train_number+valid_number),:], label[(train_number+valid_number):-1, :])
    return label, train_num, valid_num, test_num

if __name__=='__main__':
    label, train_num, valid_num, test_num = load_data()
