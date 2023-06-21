import imageio
import numpy

def conve(image):
    png_image = imageio.imwrite(numpy.zeros(0), image, format="PNG")
    return png_image