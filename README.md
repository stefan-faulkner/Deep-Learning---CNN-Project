In this project, I will be working with the CIFAR 100 dataset. The dataset contains 20 super labels of object categories,
called the coarse labels, and each of those super labels further contain categories, called the fine labels. 

There are a total of 100 fine labels in the dataset. The dataset come packaged with the Tensorflow Keras API and can be downloaded directly
via the commands below for the coarse/fine labels,

# Download CIFAR 100 with all 100 labels
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
# Doanload CIFAR 100 with the 20 superclass labels
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data(label_mode='coarse')

I will use this dataset to create a CNN based classifier to identify which images in the test set belong to which class. Furthermore, I will
will then write a report that outlines what I have done and why it was done.

