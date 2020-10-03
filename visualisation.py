from nibabel.viewers import OrthoSlicer3D

def view_image(data):
    print("Viewing data of shape", data.shape)
    OrthoSlicer3D(data).show()

