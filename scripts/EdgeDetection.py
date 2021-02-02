import os
import sys
import cv2 as cv
from scipy import ndimage, misc
from skimage import io
import numpy as np


def display_version():
    print("inside main")
    print("You are running version %s" % str(sys.version_info))
    print("Open CV version %s" % (cv.__version__))

def convert_sobel2one(image):
    sobel=ndimage.sobel(image)
    boolean_array=sobel != 0
    mono_array=boolean_array.astype(int)
    return mono_array

def convert_to_contours(image):
    im2, contours, hierarchy = cv.findContours(image, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return im2

def convert_to_mono(image):
    black_white_threshold=0
    if (image.dtype == 'float'):
        black_white_threshold=0.5
    elif (image.dtype == 'uint8'):
        black_white_threshold=128
    else:
        raise Exception("Invalid dtype %s " % (image.dtype))    

    boolean_array  = image < black_white_threshold
    mono_array = boolean_array.astype(int)
    return mono_array

def read_contours_in_file(picfile):
    print("Contour in file:%s" % picfile)
    im = cv.imread(picfile)
    imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)    
    ret, thresh = cv.threshold(imgray, 127, 255, 0)
    

    contours,result_image= cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    #Generate a White image with the same shape and dtype as original
    blank_image=np.full(imgray.shape,255, imgray.dtype)

    cv.drawContours(blank_image,contours,-1,(0,255,0),1) #last parameter decides the thickness of the line
    result=blank_image
    current_file_dir=os.path.dirname(__file__)
    outdir=os.path.join(current_file_dir,"out")
    outfile=os.path.basename(picfile)
    io.imsave(os.path.join(outdir,outfile)  , result)

def read_edge_in_file(picfile):    
    print("Edge detection in file:%s" % picfile)
    image_array=io.imread(picfile,as_gray=True)



    mono_array = convert_to_mono(image_array)
    #Follow SCIPY https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.sobel.html
    #Just conversion to Monochrome looks promising , better than sobel edge
    #Tried Contours with OPENCV, interesting results with thickness=1, but monochrome might still be better
    #
    #What next? 
    #   Line detection
    #   Circle detection
    
    #result=convert_sobel2one(mono_array)
    result=convert_to_contours(mono_array)
    result=mono_array

    current_file_dir=os.path.dirname(__file__)
    outdir=os.path.join(current_file_dir,"out")
    outfile=os.path.basename(picfile)
    io.imsave(os.path.join(outdir,outfile)  , result)

def mainloop():
    sys.path
    current_file_dir=os.path.dirname(__file__)
    path_to_smallsubset=os.path.join(current_file_dir,"../smallsubset/train")
    digit_folders=os.listdir(path_to_smallsubset)
    print(digit_folders)

    for digit_folder in digit_folders:
        absolute_digit_folder=os.path.join(path_to_smallsubset,digit_folder)
        pic_files=os.listdir(absolute_digit_folder)
        for pic_file in pic_files:            
            #read_edge_in_file(os.path.join(absolute_digit_folder,pic_file))
            read_contours_in_file(os.path.join(absolute_digit_folder,pic_file))
    
    print("to be done")

display_version()
mainloop()
