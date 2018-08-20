import cv2
import os
import hashlib
import numpy as np

def Cache(image):
    img = np.ascontiguousarray(image, dtype=np.float)
    name = hashlib.md5(img.data).hexdigest()[:8]
    if not os.path.exists("./cache/"):
        os.mkdir("./cache")
    cv2.imwrite("./cache/"+name+".png",img)