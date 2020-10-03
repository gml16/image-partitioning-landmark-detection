from nibabel.viewers import OrthoSlicer3D
import numpy as np

def view_image(data):
    data = np.squeeze(data)
    print("Viewing image with shape", np.array(data).shape)
    try:
        OrthoSlicer3D(data).show()
    except:
        print("Cannot view image, headless machine?")

