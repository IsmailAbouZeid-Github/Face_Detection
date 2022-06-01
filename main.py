from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
with open("labels.txt") as file_in:
    labels = []
    for line in file_in.readlines():

        newline=line.strip()
        newstring= ''.join([i for i in newline if not i.isdigit()])
        labels.append(newstring)
print(labels)

# Load the model
model = load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
# Replace this with the path to your image
while True:
    _,frame = cap.read()
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image= frame.copy()
    image = cv2.resize(image,size)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    humans = face_cascade.detectMultiScale(gray, 1.9, 1)

    for (x, y, w, h) in humans:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    conf = prediction.max()
    idx = np.where(prediction == conf)
    img_idx= idx[1][0]
    who = labels[img_idx]
    if conf>0.6:
        cv2.putText(frame,str(who),(100,100),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),3)
    else:
        cv2.putText(frame, "Where?", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 0), 3)

    cv2.imshow("img",frame)


    if cv2.waitKey(1) and 0xFF ==ord('q'):
        break
