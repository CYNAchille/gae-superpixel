from PIL import Image
import os
import numpy as np
from matplotlib import pyplot as plt

def load_imgs( ):
    imgset = []
    for filename in os.listdir(r"./validate"):
    #for filename in os.listdir(r"./train"):
        img = np.array(Image.open("./validate/"+filename))
        img = (img / float(img.max())).astype(np.float32)
        imgset.append(img)
    return imgset

if __name__=='__main__':
    imgset = load_imgs()
    plt.imshow(imgset[2])
    plt.title('image of shape {}'.format(imgset[2].shape))
    plt.show()

