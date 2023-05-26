import cv2
import numpy as np


def translateImg():

    image = cv2.imread('image.jpg')

    h, w = image.shape[:2]

    shiftW, shiftH = w // 4, 10

    # Translation matrix
    translation_matrix = np.float32([[1, 0, shiftW], [0, 1, shiftH]]) # type: ignore

    # Apply translation to the input image using OpenCV
    translated_image = cv2.warpAffine(
        src=image, M=translation_matrix, dsize=(w, h))

    # Save the output image using OpenCV
    cv2.imwrite('./Pgm2output/translatedImg.jpg', translated_image)


def rotateImg():
    # Read the input image
    img = cv2.imread('image.jpg')

    # Get the dimensions of the image
    height, width = img.shape[:2]

    # Define the rotation angle in degrees
    angle = 45

    # Define the scale factor for the rotation
    scale = 1

    # Calculate the rotation matrix using the cv2.getRotationMatrix2D function
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, scale)

    # Apply the rotation to the image using cv2.warpAffine
    rotated = cv2.warpAffine(img, matrix, (width, height))

    # Display the rotated image
    # cv2.imshow('Rotated Image', rotated)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save the output image using OpenCV
    cv2.imwrite('./Pgm2output/rotatedImg.jpg', rotated)


def scaleImg():

# Load the input image
    img = cv2.imread('image.jpg')

    # Get the dimensions of the input image
    h, w = img.shape[:2]

    # Calculate the nearest power of 2 that is greater than or equal to the maximum dimension of the input image
    max_dim = max(h, w)
    new_dim = 2 ** (int.bit_length(max_dim) - 1)
    if new_dim < max_dim:
        new_dim *= 2

    # Resize the input image to the new dimensions using OpenCV
    resized_img = cv2.resize(img, (new_dim, new_dim))

    # Save the output image using OpenCV
    cv2.imwrite('./Pgm2output/scaledImg.jpg', resized_img)


scaleImg()

rotateImg()

translateImg()


def getRGB():

    img = cv2.imread('image.jpg')

    b, g, r = cv2.split(img)

    print(r, g, b)


def drawImg():

    # read image
    img = cv2.imread('image.jpg')

    # Draw a circle on the image
    new_img = cv2.circle(img, (img.shape[1]//2, img.shape[0]//2), 500, (0, 0, 255), thickness=50)

    # Save the output image using OpenCV
    cv2.imwrite('./Pgm2output/circledImg.jpg', new_img)

drawImg()

def bwImg():
    image = cv2.imread('image.jpg')

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save the output image using OpenCV
    cv2.imwrite('./Pgm2output/bwImg.jpg', gray_image)

bwImg()

def flipImg():
    image = cv2.imread('image.jpg')

    # Flip the image horizontally using cv2.flip()
    flipped_img = cv2.flip(image, 0)

    # Save the output image using OpenCV
    cv2.imwrite('./Pgm2output/flippedImg.jpg', flipped_img)

flipImg()


def cropImg():
    image = cv2.imread('image.jpg')

    # Define the coordinates of the ROI
    x, y, width, height = 100, 100, 2000, 2000

    # Crop the image using numpy slicing
    cropped_image = image[y:y+height, x:x+width]
    
    # Save the output image using OpenCV
    cv2.imwrite('./Pgm2output/croppedImg.jpg', cropped_image)

cropImg()

def invertImg():
    image = cv2.imread('image.jpg')

    # Invert the colors
    inverted_image = 255 - image # type: ignore

    # Save the output image using OpenCV
    cv2.imwrite('./Pgm2output/invertedImg.jpg', inverted_image)

invertImg()



def gaussBlurImg():

    # Load the input image
    img = cv2.imread('image.jpg')

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(img, (5,5), 0)

    # Save the output
    cv2.imwrite('./Pgm2output/gaussBlurImg.jpg', blur)

gaussBlurImg()

def medianBlurImg():

    # Load the input image
    img = cv2.imread('image.jpg')

    # Apply Median filter
    median = cv2.medianBlur(img, 5)

    # Save the output
    cv2.imwrite('./Pgm2output/medianBlurImg.jpg', median)   
    
medianBlurImg()

def bilateralBlurImg():

    # Load the input image
    img = cv2.imread('image.jpg')

    # Apply Bilateral filter
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)

    # Save the output
    cv2.imwrite('./Pgm2output/bilateralBlurImg.jpg', bilateral)
    
bilateralBlurImg()

def laplaceBlurImg():
    
    # Load the input image
    img = cv2.imread('image.jpg')

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian filter
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Save the output
    cv2.imwrite('./Pgm2output/laplaceBlurImg.jpg', laplacian)

laplaceBlurImg()



def matrixValueImg():

    img = cv2.imread('image.jpg')

    # Print the NumPy array
    print(np.array(img))

matrixValueImg()


def rgbToHsv():

        
    # Load the input image
    img = cv2.imread('image.jpg')

    # Convert the image from RGB to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # Save the HSV image
    cv2.imwrite('./Pgm2output/rgbToHsv.jpg', hsv)

def rgbToLab():

    # Load the input image
    img = cv2.imread('image.jpg')

    # Convert the image from RGB to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)

    # Save the LAB image
    cv2.imwrite('./Pgm2output/rgbToLab.jpg', lab)

rgbToHsv()

rgbToLab()
