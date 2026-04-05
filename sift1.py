import cv2 as cv
import math
import numpy as np


# ----- Scale Space Extrema Detection -----
# pinpoint locations for finding features
'''
https://docs.opencv.org/4.x/da/df5/tutorial_py_sift_intro.html
Steps:
    - find the difference of gaussian
    - local extrema are potential key points
'''
def scale_space_octave(image, scale):
    # convolve image with gaussian 
    sigma = 1.6
    k = math.sqrt(2)
    scale_img = image
    octave = [image]

    for i in range(scale):
        # may need to divide diff in line 26 by k*sigma - sigma????????
        gaus_one = cv.GaussianBlur(image, (5, 5), sigmaX=sigma)
        sigma = k*sigma
        gaus_two = cv.GaussianBlur(image, (5, 5), sigmaX=sigma)
        # subtract more blurred from less blurred
        difference_gaussian = gaus_two - gaus_one
        # multiply difference of gaussians by the image
        scale_img = difference_gaussian * scale_img
        octave.append(scale_img)

    return octave

def find_extrema(octave, scale):
    # a point is an extrema  when the point is larger or smaller than its 26 neighbors in an octave
    # 
    current_img = octave[scale]
    lower_img = octave[scale - 1]
    higher_img = octave[scale + 1]

    height, width = current_img.shape
    extrema = []

    # go through all pixels in current image
    # get the 3x3 arrays in all 
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            # get 3x3 arrays
            poi = current_img[y, x]
            array = current_img[y - 1:y + 2, x - 1: x + 2]
            lower_array = lower_img[y - 1:y + 2, x - 1: x + 2]
            higher_array = higher_img[y - 1:y + 2, x - 1: x + 2]
            
            # if poi is either greater than or less than all points, add to extrema array
            if (poi >= array.max() and poi > lower_array.max() and poi > higher_array.max()) or (poi <= array.min() and poi < lower_array.min() and poi < higher_array.min()):
                extrema.append((x, y))
    
    # return extrema array
    return extrema


def scale_space_extrema(image):
    octaves = []
    extrema_temp = []    #will be an array of five arrays for each octave

    # computing octaves (only 5 levels)
    for i in range(5):
        octave = scale_space_octave(image, i)
        octaves.append(octave)

    # finding extrema (the key points) in octaves
    # scales are the size of the octave
    for octave in octaves:
        for scale in range(1, len(octave) - 2):
            extrem = find_extrema(octave, scale)
            extrema_temp.append(extrem)

    # flattening the 2D extrema list
    extrema = []
    for sublist in extrema_temp:
        for item in sublist:
            extrema.append(item)

    # drawing the keypoints on the image
    output_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for (x, y) in extrema:
        cv.circle(output_image, (x, y), 2, (255, 255, 0), -1)

    # save and return image
    cv.imwrite("Output_Data/Scale.png", output_image)
    return output_image
            

# ----- Localize Key Points -----
# this will refine the key points 
# initial extraction:
'''
Get rid of:
- poor contrast key points
- poorly localized along an edge
- use taylor series expansion of DoG
'''

# further extraction:


# ----- Orientation Assignment -----
# assign orientation to points

# ----- Key Point Descriptor -----
# describe key points

def main():
    # read image and preprocessing
    blocks_img = cv.imread('Input_Data/blocks_L-150x150.png')
    blocks_gray = cv.cvtColor(blocks_img, cv.COLOR_BGR2GRAY)

    # cv.namedWindow('Gray Image', cv.WINDOW_AUTOSIZE)
    # cv.imshow('Gray Image', blocks_gray)


    # Step 1: Scale Space Extrema Detection
    scale_space_img = scale_space_extrema(blocks_gray)
    cv.namedWindow('Scale Space Image', cv.WINDOW_AUTOSIZE)
    cv.imshow('Scale Space Image', scale_space_img)

    #Step 2: 

    #Step 3:

    #Step 4:


    cv.waitKey(0)
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()