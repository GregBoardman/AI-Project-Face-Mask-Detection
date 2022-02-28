import cv2 as cv
import os
import sys


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


faceCascade = cv.CascadeClassifier(os.path.join(cv.haarcascades, 'haarcascade_frontalface_default.xml'))

video_capture = cv.VideoCapture(0)

flName = "C:\\Users\\RoiDe\\PycharmProjects\\facecut\\gregface\\mask\\mask"
nameCounter = 0

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
        cropped = cropper(frame, x, y, w, h, 0.3)
        cv.imwrite((flName + f"{nameCounter}" + ".jpg"), cropped)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        nameCounter += 1

    # Display the resulting frame
    cv.imshow('Video', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv.destroyAllWindows()
