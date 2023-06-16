from deepface import DeepFace
from mtcnn.mtcnn import MTCNN
from IPython.display import clear_output

import cv2
import os
import numpy as np

import time
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

clear_output()


def load_img(path):
    """
    Function:
      - Loads the image from path
      - Crops the face out of the image using MTCNN()
      - Saves image as an np array

    Args:
      -path: of the image

    Output:
      -returns the cropped image in the form of numpy array
  """

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print("chkpt1")

    detector = MTCNN()
    results = detector.detect_faces(img)
    clear_output()

    margin = 0  # Decrease this value to reduce the margin around the face
    padding = (30, 30, 30, 30)

    x, y, w, h = results[0]['box']

    x -= margin
    y -= margin
    w += margin * 2
    h += margin * 2

    # Apply padding to the coordinates
    x -= padding[0]
    y -= padding[1]
    w += padding[0] + padding[2]
    h += padding[1] + padding[3]

    img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 30)
    face = img[y:y + h, x:x + w]
    img = np.array(face)

    img = (img / 255.0).astype(np.float32)

    # print("chkpt2")
    return img


imag = load_img("/content/College_id.jpg")
plt.imshow(imag, interpolation="nearest")
plt.show()

# img1 = load_img("/content/PAN.jpg")
img2 = load_img("/content/moustache.png")
plt.imshow(img2, interpolation="nearest")
plt.show()

verification = DeepFace.verify(imag, img2, enforce_detection=False)
time.sleep(5)

clear_output()

verif = verification["verified"]

print(verif)

accuracy = verification["distance"]

if verif:
    print((1 - verification["distance"]))
else:
    if accuracy > 0.48:
        print("Please try removing any facial coverings or try a newer image")
    # elif accuracy > 0.45:
    #   print("Try removing any background noise and removing facial coverings")
    else:
        print(verification['distance'])

print(f"Estimated accuracy: {1 / (1 + accuracy)}")
