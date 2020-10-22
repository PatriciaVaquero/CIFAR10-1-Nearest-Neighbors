import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
import time
tic = time.process_time()

def class_acc(pred, gt):
    correct = 0
    for i in range(len(gt)):
        if gt[i] == pred[i]:
            correct += 1
    return correct / float(len(gt)) * 100.0

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding="latin1")
    return dict

def classifier_random(x):
    randomlist = []
    print(x)
    for i in range(x.shape[0]):
        n = random.randint(0, 9)
        randomlist.append(n)
    return np.array(randomlist)

def cifar10_classifier_1nn(x, trdata, trlabels):
    dist = np.zeros(len(trdata))
    for i in range(0, len(trdata)):
        dist[i] = np.sum(np.subtract(trdata[i], x) ** 2)
    test_label = trlabels[dist.argmin()]
    return test_label

# datadict = unpickle('C:/Users/patir/OneDrive/Documentos/cifar-10-batches-py/data_batch_1')
Training_files = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
data = []
labels = []
for i in range(5):
    raw_data = unpickle('cifar-10-batches-py/' + Training_files[i])
    data.append(raw_data["data"])
    labels.append(raw_data["labels"])
train_images = np.concatenate(data)
train_images = train_images.astype('int32')
train_classes = np.concatenate(labels)

labeldict = unpickle('cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

# Ejercicio 2
print(class_acc(train_classes, train_classes))

# Ejercicio 3
pred_rand = classifier_random(train_images)
print(pred_rand)
accuracy_result = class_acc(pred_rand, train_classes)
print(accuracy_result)

# Ejercicio 4
datadict = unpickle('cifar-10-batches-py/test_batch')
X = datadict["data"]
Y = datadict["labels"]
test_images = X
test_images = test_images.astype('int32')
test_classes = np.array(Y)
labeldict = unpickle('cifar-10-batches-py/batches.meta')
label_names = labeldict["label_names"]

x_label = []
for x in range(0, 10000):
    # test_labels[x] = cifar10_classier_1nn(test_images[x], train_images,train_classes)
    x_label.append(cifar10_classifier_1nn(test_images[x], train_images, train_classes))
    print(x_label)
print(class_acc(x_label, test_classes))
toc = time.process_time()
print(toc - tic)