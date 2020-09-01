
#importing the keras library and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import matplotlib.pyplot as plt
import os
import h5py
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img
import PIL

#inatializing the CNN
classifier=Sequential()

#step-1:convolution layer                                                                                                    
classifier.add(Convolution2D(64, (3, 3), border_mode='same', input_shape=(50, 50, 3), activation='relu')) 
classifier.add(Convolution2D(64, 3, 3,activation='relu')) 
classifier.add(MaxPooling2D(pool_size=(2,2)))

#add another convolution layer
classifier.add(Convolution2D(128, 3, 3,activation='relu')) 
classifier.add(Convolution2D(128, 3, 3,activation='relu')) 
classifier.add(MaxPooling2D(pool_size=(2,2)))
#add another convolution layer
classifier.add(Convolution2D(256, 5, 5,activation='relu')) 
classifier.add(Convolution2D(256, 5, 5,activation='relu')) 
classifier.add(MaxPooling2D(pool_size=(2,2)))
#step-3:Flattering
classifier.add(Flatten())
 
#step-4:full connection
classifier.add(Dense(output_dim=512, activation='relu'))
classifier.add(Dense(output_dim=256, activation='relu'))
classifier.add(Dense(output_dim=36, activation='softmax'))

#compile the CNN
classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
print (classifier.summary())
#Fitting the CNN model to image
from keras.preprocessing.image import ImageDataGenerator 
train_datagen = ImageDataGenerator(rescale=1./255,
                               #shear_range=0.2,
                               #zoom_range=0.2,
                               #horizontal_flip=True
                               )

test_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory('Train_data/train',
                                                target_size=(50, 50),
                                                batch_size=200,
                                                class_mode='categorical')
 
test_set = test_datagen.flow_from_directory('Train_data/val',
                                            target_size=(50, 50),
                                            batch_size=200,
                                            class_mode='categorical')

history = classifier.fit_generator(train_set,
                        samples_per_epoch=6000,
                        nb_epoch=15,
                        validation_data=test_set,
                        nb_val_samples=1000)


#save the multidimentional arrays data into MNIST.h5 file
classifier.save('sign.h5')

# #Test on the image form test folders
# test_model = load_model('sign.h5')

# img = load_img('testSample\img_83.jpg',False,target_size=(28,28))
# x = img_to_array(img)
# x = np.expand_dims(x, axis=0)
# preds = test_model.predict_classes(x)
# prob = test_model.predict_proba(x)
# print(preds, prob)

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()