import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import os.path
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import cv2 as cv
import os
import glob
import sys
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#Method for cropping images
def cropper(cropper_image, cropper_x, cropper_y, cropper_width, cropper_height, scale=1):
    # following code snipit ALPHA is from
    # https://stackoverflow.com/questions/56331959/scale-and-crop-an-image-in-python-pil-without-exceeding-the-image-dimensions
    # Quick note, I am the user Fritz from this link.

    # format for easier use with crop function
    top: int = cropper_y + cropper_height
    left: int = cropper_x
    bottom: int = cropper_y
    right: int = cropper_x + cropper_width

    # calc the center
    crop_center_x = int(left + cropper_width / 2)
    crop_center_y = int(top - cropper_height / 2)

    # calculate max values
    max_width = int(min(crop_center_x, (gray.shape[1] - crop_center_x)))
    max_height = int(min(crop_center_y, (gray.shape[0] - crop_center_y)))

    # make crop area bigger by ratio
    new_width = int(cropper_width + (cropper_width * scale))

    # Here we are using the previously calculated value for max_width to
    # determine if the new one would be too large.
    if max_width < new_width / 2:
        new_width = int(2 * max_width)

    new_height = int(cropper_height + (cropper_height * scale))

    # Do the same for the height, update width if necessary
    if max_height < new_height / 2:
        new_height = int(2 * max_height)

    new_left = int(crop_center_x - new_width / 2)
    new_right = int(crop_center_x + new_width / 2)
    new_bottom = int(crop_center_y - new_height / 2)
    new_top = int(crop_center_y + new_height / 2)

    # end of code snipit ALPHA

    return cropper_image[new_bottom:new_top, new_left:new_right]

#data generator used from https://towardsdatascience.com/image-detection-from-scratch-in-keras-f314872006c9
#ImageDataGenerator Used to take image data from a directory and reshape it to be used by the model.
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
#generator for the validation data
test_datagen = ImageDataGenerator(rescale=1. / 255)
#Training Image Set
train_generator = train_datagen.flow_from_directory(
    "INSERT ABSOLUTE DIRECTORY PATH HERE", target_size=(150, 150), batch_size=20, class_mode='binary')
#Validation Image Set
validation_generator = test_datagen.flow_from_directory(
    "INSERT ABSOLUTE DIRECTORY PATH HERE", target_size=(150, 150), batch_size=20, class_mode='binary')

#convolution neural network from https://www.pluralsight.com/guides/image-classification-using-tensorflow designed by Vaibhav Sharma
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    keras.layers.MaxPool2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')])

model.summary()

#If model is not saved uncomment this to save the modle so it does not need to be trained every time
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['acc'])
#
#Model is trained for 10 epochs and then tested on the validation data every epoch in order to prevent overfitting.
# maskmodel = model.fit_generator(train_generator, epochs=10, validation_data=validation_generator)
# model.save('masks.h5')

#load a saved model
model.load_weights('masks.h5')


#begin cam code

faceCascade = cv.CascadeClassifier(os.path.join(cv.haarcascades, 'haarcascade_frontalface_default.xml'))

video_capture = cv.VideoCapture(0)

#real time face capture using OpenCV.   https://github.com/shantnu/Webcam-Face-Detect used to create
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # find faces
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        # crop image
        cropped = cropper(frame, x, y, w, h, 0.30)

        # Plug cropped image into detector
        inputCropped = cv.cvtColor(cropped,cv.COLOR_BGR2RGB)
        inputCropped = cv.resize(inputCropped, (150, 150))
        inputCropped = inputCropped / 255
        inputCropped = inputCropped.reshape((1,) + inputCropped.shape)


        # Create green rectangle if mask
        print(f"Nomask likelyhood: {model.predict(inputCropped)}")
        if model.predict(inputCropped) <= 0.5:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #qprint("1mask")
        # Create Red rectangle if no mask
        if model.predict(inputCropped) > 0.5:
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            #print("nomask")
        # Create Blue Rectangle if unsure
        #if model.predict(inputCropped) > 0.3 and model.predict(inputCropped) < 0.7:
        #    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #    #print("2mask")

    # Display the resulting frame
    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv.destroyAllWindows()
