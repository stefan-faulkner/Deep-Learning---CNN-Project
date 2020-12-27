#!/usr/bin/env python
# coding: utf-8

# ## Student Details

# In[ ]:


#Student_Id: 2514007F
#Large Scale Computing for Data Analytics Project


# ## Importing Necessary Libraries

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import tensorboard
from datetime import datetime

np.set_printoptions(suppress=True)


# In[ ]:


#from google.colab import drive
#drive.mount('/content/drive',force_remount=True)

#%cd '/content/drive/My Drive/CNN Project (LSC)'


# # Part 1 of the Project 

# ## Loading Dataset and set-up the labels

# In[ ]:


# Download CIFAR 100 with the 20 superclass labels
(x_train, y_train), (x_test, y_test) =    tf.keras.datasets.cifar100.load_data(label_mode='coarse')

labels = ['aquatic mammals','fish','flowers','food containers','fruit and vegetables','household electrical devices',
    'household furniture','insects','large carnivores','large man-made outdoor things','large natural outdoor scenes',
    'large omnivores and herbivores','medium-sized mammals','non-insect invertebrates','people','reptiles','small mammals',
    'trees','vehicles 1','vehicles 2']


# ## Now Preprocessing our data 
# 

# In[ ]:


y_input = tf.keras.utils.to_categorical(y_train)
x_input = (x_train/255.0).astype(np.float32) # Normalizing our data

y_valid = tf.keras.utils.to_categorical(y_test)
x_valid = (x_test/255.0).astype(np.float32) # Normalizing our data


# In[ ]:


# Note that we can get the shape of the tensor with the shape property

print('Shape of training and test data ')
print ('x_input = ' + str(x_input.shape))
print ('y_input = ' + str(y_input.shape))


print ('x_valid  = ' + str(x_valid.shape))
print ('y_valid  = ' + str(y_valid.shape))


# ## Visualizing some random images before we build our CNN based Classifer Model

# In[ ]:


# Setting up grid structure 

fig, axs = plt.subplots(2,3, figsize=(12, 12))
fig.subplots_adjust(hspace = .2, wspace=.2)
axs = axs.ravel()

# plot 6 images at random
for i in range(6):
    # choose a random image
    j=np.random.randint(0,len(x_train))

    label = y_train[j]
    image = x_train[j]

    # show the image and display its category
    axs[i].imshow(image)
    axs[i].set_title(labels[int(label)])

    # turn off grids and axis labels
    axs[i].grid(False)
    axs[i].get_xaxis().set_visible(False)
    axs[i].get_yaxis().set_visible(False)
plt.show()


# ## Builiding our CNN based Classifier Model with the architecture provided to us in the project description

# In[ ]:


#Creating the sequential model
model1 = tf.keras.Sequential()

#Below I will walk through the architecture step by step which we were given to use 

#Starting off with the first layer, this should be a convolution layer with a 3x3 kernel size and 32 filters.
model1.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation=tf.nn.relu,padding='same', input_shape=( 32, 32, 3)))

#The next layer should be a max pooling layer with a pool size of 2x2.
model1.add(tf.keras.layers.MaxPool2D(pool_size=(2,2)))

#The next layer should be a convolution layer with a 3x3 kernel size and 64 filters.
model1.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), activation=tf.nn.relu,padding='same', input_shape=( 32, 32, 3)))

#Flattening
model1.add(tf.keras.layers.Flatten())

#The next layer (after flattening) should be a dense layer with 128 units.
model1.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))

#The final layer should be a dense layer with the softmax activation function.
model1.add(tf.keras.layers.Dense(units=20,activation=tf.nn.softmax))


# In[ ]:


#Now we are going to Compile and Train the model

#The Adam Optimizer  will be the one which is used 
#Our learning rate will be 0.0001

lr=0.0001

model1.compile(optimizer=tf.keras.optimizers.Adam(lr),loss='categorical_crossentropy',metrics=['accuracy'])

# create a callback that will stop training if the validation loss hasn't improved for 3 epochs
logdir = 'tflogs'
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
             tf.keras.callbacks.TensorBoard(log_dir=logdir)]


model1.fit(x_input, y_input, epochs=100,
    batch_size=512,
    callbacks=callbacks,
    validation_data=(x_valid, y_valid))


# In[ ]:


model1.summary()


# ## Visualizing in Tensorboard

# In[ ]:


get_ipython().run_line_magic('load_ext', 'tensorboard')
get_ipython().run_line_magic('tensorboard', '--logdir tflogs')


# ## Now Let's assess the Accuracy

# In[ ]:


# Looking at the accuracy by making predictions for the test data:

predict = model1.predict(x_test/255.0).astype(np.float32)
y_pred = np.argmax(predict,axis=-1)
print('Test accuracy: ', np.sum(y_pred==y_test[:,0])/len(y_test))

# Note that we are getting a test accuracy of approximately 47%


# # Part 2 of the Project 

# ## Altering our CNN Based Classifier Model Architecture to try and increase the accuracy

# In[ ]:


# Creating Model
model2 = tf.keras.Sequential()
model2.add(tf.keras.layers.Conv2D(kernel_size=(3,3), filters=32, activation='relu'))
model2.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model2.add(tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, activation='relu'))
model2.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model2.add(tf.keras.layers.Conv2D(kernel_size=(3,3), filters=128, activation='relu'))
model2.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model2.add(tf.keras.layers.Flatten())
model2.add(tf.keras.layers.Dense(units=256, activation='relu'))
model2.add(tf.keras.layers.Dense(units=256, activation='relu'))
model2.add(tf.keras.layers.Dense(units=512, activation='relu'))
model2.add(tf.keras.layers.Dense(units=128, activation='relu'))
model2.add(tf.keras.layers.Dense(units=20, activation=tf.nn.softmax))






# In[ ]:


# Now Compiling the Model

lr=0.001
model2.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])

# create a callback that will stop training if the validation loss hasn't improved for 3 epochs
logdir='tflogs2'
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
             tf.keras.callbacks.TensorBoard(log_dir=logdir)]

# Next training the model
model2.fit(x_input, y_input, epochs=100,
    batch_size=50,
    callbacks=callbacks,
    validation_data=(x_valid, y_valid))


# ### Visualizing in Tensorboard

# In[ ]:



get_ipython().run_line_magic('tensorboard', '--logdir tflogs2')


# ## Now Let's Optimize the  hyperparameters for our CNN Based Classifier Model Architecture above

# In[ ]:


# Creating Model
model3 = tf.keras.Sequential()
model3.add(tf.keras.layers.Conv2D(kernel_size=(3,3), filters=32, activation='relu'))
model3.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model3.add(tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, activation='relu'))
model3.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model3.add(tf.keras.layers.Conv2D(kernel_size=(3,3), filters=128, activation='relu'))
model3.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model3.add(tf.keras.layers.Flatten())
model3.add(tf.keras.layers.Dense(units=256, activation='relu'))
model3.add(tf.keras.layers.Dense(units=256, activation='relu'))

model3.add(tf.keras.layers.Dense(units=256, activation='relu'))

model3.add(tf.keras.layers.Dense(units=512, activation='relu'))

model3.add(tf.keras.layers.Dense(units=20, activation=tf.nn.softmax))


# Now Compiling the Model

lr=0.001
model3.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])

# create a callback that will stop training if the validation loss hasn't improved for 3 epochs
logdir='tflogs3'
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2),
             tf.keras.callbacks.TensorBoard(log_dir=logdir)]


# Training the model
model3.fit(x_input, y_input, epochs=100,
    batch_size=128,
    callbacks=callbacks,
    validation_data=(x_valid, y_valid))



# ### Visualizing in Tensorboard

# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir tflogs3')


# ### Now Let's assess the Accuracy

# In[ ]:


# Looking at the accuracy by making predictions for the test data:

predict = model3.predict(x_test/255.0).astype(np.float32)
y_pred = np.argmax(predict,axis=-1)
print('Test accuracy: ', np.sum(y_pred==y_test[:,0])/len(y_test))

# Note that we are getting a test accuracy of approximately 52%


# # Part 3 of the Project

# ## We start off by downloading the CIFAR-100 dataset with fine labels

# In[ ]:


(x_train_100, y_train_100), (x_test_100, y_test_100) = tf.keras.datasets.cifar100.load_data()

y_input_100 = tf.keras.utils.to_categorical(y_train_100)
x_input_100 = (x_train_100/255.0).astype(np.float32)

y_valid_100 = tf.keras.utils.to_categorical(y_test_100)
x_valid_100 = (x_test_100/255.0).astype(np.float32)


# ## Setting up Model Architecture to Train Data

# In[ ]:


#Creating Model
model4 = tf.keras.Sequential()
model4.add(tf.keras.layers.Conv2D(kernel_size=(3,3), filters=32, activation='relu'))
model4.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model4.add(tf.keras.layers.Conv2D(kernel_size=(3,3), filters=64, activation='relu'))
model4.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model4.add(tf.keras.layers.Conv2D(kernel_size=(3,3), filters=128, activation='relu'))
model4.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model4.add(tf.keras.layers.Flatten())
model4.add(tf.keras.layers.Dense(units=256, activation='relu'))
model4.add(tf.keras.layers.Dense(units=256, activation='relu'))

model4.add(tf.keras.layers.Dense(units=256, activation='relu'))
model4.add(tf.keras.layers.Dense(units=512, activation='relu'))
model4.add(tf.keras.layers.Dense(units=100, activation=tf.nn.softmax))  #FINAL DENSE LAYER

#Compiling the model
lr=0.001
model4.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])

# create a callback that will stop training if the validation loss hasn't improved for 3 epochs
logdir='tflogs4'
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
             tf.keras.callbacks.TensorBoard(log_dir=logdir)]

#Training the model
model4.fit(x_input_100, y_input_100, epochs=100,
    batch_size=128,
    callbacks=callbacks,
    validation_data=(x_valid_100, y_valid_100))


# ### Visualizing in Tensorboard

# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir tflogs4')


# ### Now Let's assess the Accuracy

# In[ ]:


predict = model4.predict(x_test_100/255.0).astype(np.float32)
y_pred = np.argmax(predict,axis=-1)
print('Test accuracy: ', np.sum(y_pred==y_test_100[:,0])/len(y_test_100))

# Note that we are getting a test accuracy of 37%


# ## Now we are going to do Transfer Learning using a pretrained VGG16 network

# In[ ]:


from keras.applications import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.callbacks import TensorBoard
from keras.optimizers import Adam


# In[ ]:


#VGG16 MODEL
vgg_model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

#Freeze the layers:
for layer in vgg_model.layers[:]:
  layer.trainable = False

#Check the trainable status:
for layer in vgg_model.layers:
  print(layer, layer.trainable)

#Create a new model:
model5 = tf.keras.Sequential()

#Add the VGG
model5.add(vgg_model)

#Add the fully-connected layers 
model5.add(tf.keras.layers.Flatten(name="flatten"))
model5.add(tf.keras.layers.Dense(units=512, activation=tf.nn.relu))
model5.add(tf.keras.layers.Dense(units=100, activation=tf.nn.softmax))


# In[ ]:


#Compiling the model
lr=0.001
model5.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])

# create a callback that will stop training if the validation loss hasn't improved for 3 epochs
logdir='tflogs5'
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
             tf.keras.callbacks.TensorBoard(log_dir=logdir)]

#Training the model
model5.fit(x_input_100, y_input_100, epochs=100,
    batch_size=128,
    callbacks=callbacks,
    validation_data=(x_valid_100, y_valid_100))


# ### Visualizing in Tensorboard

# In[ ]:


get_ipython().run_line_magic('tensorboard', '--logdir tflogs5')


# ### Now Let's assess the Accuracy

# In[ ]:


predict = model5.predict(x_test_100/255.0).astype(np.float32)
y_pred = np.argmax(predict,axis=-1)
print('Test accuracy: ', np.sum(y_pred==y_test_100[:,0])/len(y_test_100))

# Note that we are getting a test accuracy of 38%


# In[ ]:




