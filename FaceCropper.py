import cv2 as cv
import os
import glob
# import matplotlib.pyplot as plt
import numpy as np


# following code snipit BETA is from
# https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
def auto_canny(img, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(img)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv.Canny(img, lower, upper)
    # return the edged image
    return edged
# end of code snipit BETA


def cropper(cropper_image, cropper_x, cropper_y, cropper_width, cropper_height, scale):
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


##################################################
# BEGIN MAIN
##################################################

# face detector
faceCascade = cv.CascadeClassifier(os.path.join(cv.haarcascades, 'haarcascade_frontalface_default.xml'))

# put path inputs and outputs here
pathIN = ['C:\\Users\\RoiDe\\PycharmProjects\\facecut\\data\\train\\nomasks',
          'C:\\Users\\RoiDe\\PycharmProjects\\facecut\\data\\train\\masks',
          'C:\\Users\\RoiDe\\PycharmProjects\\facecut\\data\\test\\nomasks',
          'C:\\Users\\RoiDe\\PycharmProjects\\facecut\\data\\test\\masks']

pathOUT = ['C:\\Users\\RoiDe\\PycharmProjects\\facecut\\data2\\train\\nomasks',
           'C:\\Users\\RoiDe\\PycharmProjects\\facecut\\data2\\train\\masks',
           'C:\\Users\\RoiDe\\PycharmProjects\\facecut\\data2\\test\\nomasks',
           'C:\\Users\\RoiDe\\PycharmProjects\\facecut\\data2\\test\\masks']

filename = ['data2_train_nomask_pic',
            'data2_train_mask_pic',
            'data2_test_nomask_pic',
            'data2_test_mask_pic']


for pathinput, pathoutput, flname in zip(pathIN, pathOUT, filename):
    # for future file naming
    nameCounter = 1
    fileName = "output"

    # change current directory to input
    os.chdir(pathinput)

    for n in glob.glob("*.jpg"):
        # Get images one at a time
        image = cv.imread(pathinput + "\\" + n)
        # convert to a gray image
        try:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        except:
            print("This Image Failed to load: " + n)
            print("Please rename this image.")
            continue

        # find faces
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,  # original is 1.1
            minNeighbors=5,  # original is 5
            minSize=(30, 30),  # original is 30, 30
            flags=cv.CASCADE_SCALE_IMAGE
        )

        # change current directory to the output file
        os.chdir(pathoutput)

        # crop for face
        for f in faces:
            # get region
            x, y, crop_width, crop_height = [v for v in f]

            # create edged out image
            #edge = auto_canny(img=gray, sigma=0.33)

            # crop edged out image
            cropped_image = cropper(
                                cropper_image=gray,
                                cropper_x=x,
                                cropper_y=y,
                                cropper_width=crop_width,
                                cropper_height=crop_height,
                                scale=0.20
                                )

            #edge the crop after cropping instead before cropping
            edge = auto_canny(img=cropped_image, sigma=0.33)

            # store image
            cv.imwrite((flname + f"{nameCounter}" + ".jpg"), edge)

            # increase name counter
            nameCounter += 1
        # END OF FACE LOOP
    # END OF PATH LIST LOOP
