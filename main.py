import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from collections import defaultdict
import tensorflow as tf


def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    # cifar_train_data_dict
    # 'batch_label': 'training batch 5 of 5'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    # test data
    # cifar_test_data_dict
    # 'batch_label': 'testing batch 1 of 1'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
        cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names


if __name__ == "__main__":

    cifar_10_dir = '/Users/soumilchugh/Downloads/cifar-10-batches-py'

    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = load_cifar_10_data(cifar_10_dir)

    print("Train data: ", train_data.shape)
    print("Train filenames: ", train_filenames.shape)
    print("Train labels: ", train_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test filenames: ", test_filenames.shape)
    print("Test labels: ", test_labels.shape)
    print("Label names: ", label_names.shape)

    def maxpool2d(inputData):
        return tf.compat.v1.nn.max_pool(value = inputData, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')

    weightsDict = defaultdict()

    def conv2d(inputFeature,filterSize, inputSize, outputSize, name,strides = 1):
        filter_shape = [filterSize, filterSize, inputSize, outputSize]
        weightName = name + "_W"
        biasName = name + "_b"
        with tf.compat.v1.variable_scope("variable", reuse=tf.compat.v1.AUTO_REUSE):
            weightsDict[weightName] = tf.compat.v1.get_variable(weightName, shape=filter_shape, dtype = tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            weightsDict[biasName] = tf.compat.v1.get_variable(biasName, shape = outputSize,dtype = tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        convOutput = tf.compat.v1.nn.conv2d(input = inputFeature, filter = (weightsDict[weightName]), strides=[1, strides, strides, 1], padding='SAME', name = name)
        finalOutput = tf.compat.v1.nn.bias_add(convOutput, (weightsDict[biasName]))
        conv = tf.nn.relu(finalOutput, name=name)
        return conv


    def fcBlock(currentInput, inputshape, outputShape, name):
        filter_shape = [inputshape, outputShape]
        weightName = name + "_W"
        biasName = name + "_b"
        with tf.compat.v1.variable_scope("variable", reuse=tf.compat.v1.AUTO_REUSE):
            weightsDict[weightName] = tf.compat.v1.get_variable(weightName, shape=filter_shape, dtype = tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            weightsDict[biasName] = tf.compat.v1.get_variable(biasName, shape = outputShape, dtype = tf.float32, initializer=tf.contrib.layers.xavier_initializer())
        fc1 = tf.matmul(currentInput,weightsDict[weightName]) +  weightsDict[biasName]
        return fc1

    restore = False
    with tf.Session() as sess:

        data = tf.placeholder(tf.float32, [None, 32, 32, 3])
        label = tf.placeholder(tf.int64, [None])
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            newdata = data-mean
        conv1 = conv2d(newdata,3,3,64,name = "conv1_1")
        conv2 = conv2d(conv1,3,64,64,name = "conv1_2")
        pool1 = maxpool2d(conv2)
        conv3 = conv2d(pool1,3,64,128,name = "conv2_1")
        conv4 = conv2d(conv3,3,128,128,name = "conv2_2")
        pool2 = maxpool2d(conv4)
        conv5 = conv2d(pool2,3,128,256,name = "conv3_1")
        conv6 = conv2d(conv5,3,256,256,name = "conv3_2")
        conv7 = conv2d(conv6,3,256,256,name = "conv3_3")
        # Can use the remaining layers as well
        '''
        pool3 = maxpool2d(conv7)
        conv8 = conv2d(pool3,3,256,512,name = "conv4_1")
        conv9 = conv2d(conv8,3,512,512,name = "conv4_2")
        conv10 = conv2d(conv9,3,512,512,name = "conv4_3")
        
        pool4 = maxpool2d(conv10)
        conv11 = conv2d(pool4,3,512,512,name = "conv5_1")
        conv12 = conv2d(conv11,3,512,512,name = "conv5_2")
        conv13 = conv2d(conv12,3,512,512,name = "conv5_3")
        '''
        pool5 = tf.reduce_mean(conv7, axis=[1,2])
        #flatten = tf.compat.v1.layers.flatten(conv10)
        fc2 = fcBlock(pool5,256,10,name = "fc1")
        #fc1Activation = tf.nn.relu(fc,name = 'relu')
        #fc2 = fcBlock(fc1Activation,256,10,name = "fc2")
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = label, logits = fc2, name=None))
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01)
        #optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate = 0.01, momentum = 0.9, use_locking=False, name='Momentum', use_nesterov=False)
        optimize = optimizer.minimize(loss,var_list = [weightsDict["fc1_W"],weightsDict["fc1_b"]])
        acc, acc_op = tf.metrics.accuracy(labels=label, predictions=tf.argmax(tf.nn.softmax(fc2),1))
        init = tf.compat.v1.global_variables_initializer()
        init_l = tf.compat.v1.local_variables_initializer()
        sess.run(init)
        sess.run(init_l)
        print("Initialisation completed")
        if restore:
            saver = tf.train.Saver()
            saver.restore(sess, "model.ckpt")
        #flops = tf.compat.v1.profiler.profile(tf.get_default_graph(),options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation())
        #print('FLOP = ', flops.total_float_ops)
        weights = np.load("vgg16_weights.npz")
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if k in weightsDict:
                weightsDict[k].assign(weights[k])
                #print i, k, np.shape(weights[k])
        epochs = 20
        batchSize = 32
        testbatches = int(test_data.shape[0]/batchSize)
        for i in range(epochs):
            trainingError = []
            trainingAccuracy = []
            trainbatches = int(train_data.shape[0]/batchSize)
            print ("Batches ", trainbatches)
            dataset = tf.data.Dataset.from_tensor_slices((train_data,train_labels))
            dataset = dataset.shuffle(train_data.shape[0])
            dataset = dataset.batch(batchSize)
            iterator = dataset.make_initializable_iterator()
            images, labels = iterator.get_next()
            sess.run(iterator.initializer)
            for j in range(trainbatches):
                inputData, labelData = sess.run([images,labels])
                _, lossValue = sess.run([optimize,loss], feed_dict={data:inputData,label:labelData})
                #print ("Loss is", lossValue)
                trainingError.append(lossValue)
                accuracy = sess.run(acc_op, feed_dict = {label:labelData,data:inputData})
                trainingAccuracy.append(accuracy)
                #print('Accuracy: ', accuracy)
                #print (sess.run(tf.argmax(tf.nn.softmax(fc2),1), feed_dict = {label:labelData,data:inputData}))
            print ("Training Error", np.average(trainingError))
            print ("Training Accuracy", np.average(trainingAccuracy))
        testDataset = tf.data.Dataset.from_tensor_slices((test_data,test_labels))
        testDataset = testDataset.shuffle(train_data.shape[0]).batch(batchSize)
        testiterator = testDataset.make_initializable_iterator()
        images, labels = testiterator.get_next()
        sess.run(testiterator.initializer)
        testaccuracy = []
        for j in range(testbatches):
            inputData, labelData = sess.run([images,labels])
            accuracy = sess.run(acc_op, feed_dict = {label:labelData,data:inputData})
            print('Accuracy: ', accuracy)
            testaccuracy.append(accuracy)
        print ("Test Accuracy ", np.average(testaccuracy))
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(sess,"model.ckpt")

    