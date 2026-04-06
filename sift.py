import cv2 as cv
import math
import numpy as np


# ----- Scale Space Extrema Detection -----
# pinpoint locations for finding features

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
    return output_image, extrema, octaves
            

# ----- Localize Key Points -----
# this will refine the key points 
# go through selected extrema and if they meet the criteria, they are stored in new extrema list

# initial extraction:
def initial_extraction(image, extrema, dog, threshold=20):
    new_extrema = []
    for (x, y) in extrema:
        if abs(image[y, x]) > threshold:
            new_extrema.append((x, y))  #only adding high contrast pixels

    return new_extrema

# further extraction:
def further_extraction(extrema, dog, threshold=10):
    # compute hessian matrix
    # need thres and det to find r of point
    # eliminate point if r > threshold
    # for calulation of r, only Dxx, Dyy, and Dxy parts of the Hessian matrix are needed
    # for each of these values, approximate the second derivative
    new_extrema = []
    for (x, y) in extrema:
        poi = dog[y, x]
        Dxx = dog[y, x + 1] + dog[y, x - 1] - 2*poi
        Dyy = dog[y + 1, x] + dog[y - 1, x] - 2*poi
        Dxy = (dog[y + 1, x + 1] - dog[y - 1, x + 1] - dog[y + 1, x - 1] + dog[y - 1, x - 1])/4

        thres = Dxx + Dyy
        det = Dxx*Dyy - Dxy**2

        if det >= 0 and (thres**2/det < threshold):
            new_extrema.append((x, y))

    return new_extrema


def key_point_localization(image, extrema, octaves):
    new_extrema = []
    dog = octaves[1][1] #get a DoG image from the second octave

    # initial extraction
    ie_extrema = initial_extraction(image, extrema, dog)
    ie_img = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for (x, y) in ie_extrema:
        cv.circle(ie_img, (x, y), 2, (255, 255, 0), -1)
    cv.imwrite("Output_Data/IEKPLoc.png", ie_img)
    

    # further extraction
    new_extrema = further_extraction(ie_extrema, dog)

    # drawing the keypoints on the image
    output_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    for (x, y) in new_extrema:
        cv.circle(output_image, (x, y), 2, (255, 255, 0), -1)

    cv.imwrite("Output_Data/KPLoc.png", output_image)
    return output_image, new_extrema


# ----- Orientation Assignment -----
# assign orientation to points
# weighted direction histogram
def find_orientation_magnitude(image):
    image1 = image.astype(np.float32)

    # the central derivatives for horizontal (dx) and vertical change(dy); done with the sobel operator
    dx= cv.Sobel(image1, cv.CV_32F, dx=1, dy=0, ksize=3)
    dy = cv.Sobel(image1, cv.CV_32F, dx=0, dy=1, ksize=3)

    # magnitude and orientation
    mag = np.sqrt(dx**2 - dy**2)
    ori = np.degrees(np.arctan(dy, dx)) % 360   #default radian, convert to degrees

    return mag, ori

# MAIN
def orientation_assignment_keypoints(image, keypoints):
    # Using 8x8 area for orientation assignment
    # keep in range of a circle surrounding the 8x8 area
    hist_array = np.zeros(36)   #array serves as a 36 bin histogram, with each bin being 10 degrees (index is degrees)
    new_keypoints = []
    magnitude, orientation = find_orientation_magnitude(image)
    mag_height, mag_width = magnitude.shape

    for (x, y) in keypoints:
        # go through a 8x8 area around the keypoint
        for h in range(x - 3, x + 4):
            for k in range(y - 3, y + 4):
                # only calculate for items in the 8x8 area
                if h < 0 or k < 0 or h >= mag_width or k >= mag_height:
                    continue

                # magnitude and orientation for the region
                mag = magnitude[k, h]
                ori = orientation[k, h] 
                if math.isnan(mag) or math.isnan(ori):
                    continue

                i = int(ori//10)
                # print(i)

                # add to the total magnitude of the array at the right orientation
                hist_array[i] += mag

        print(hist_array)
        dominant_ori = np.argmax(hist_array)
        dominant_angle = dominant_ori*10
        dominant_mag = max(hist_array)

        new_keypoints.append((x, y, dominant_angle, dominant_mag))  #only append the key point with the dominant magnitude in the region

    return new_keypoints


def orientation_assignment(image, keypoints):
    ori_key = orientation_assignment_keypoints(image, keypoints)

    new_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    image1 = new_image.astype(np.float32)
    for (x, y, orientation, magnitude) in ori_key:
        # print(orientation)
        x2 = int(x + 10*np.cos(orientation))
        y2 = int(y + 10*np.sin(orientation))
        cv.arrowedLine(new_image, (x, y), (x2, y2), (0, 0, 255))

    cv.imwrite("Output_Data/OrAs.png", new_image)

    return new_image, ori_key


# ----- Key Point Descriptor -----
# describe key points
# using 16x16 area for descriptor
def kp_descriptors(keypoints, magnitude, orientation):
    descriptors = []
    
    for (x, y, ori, mag) in keypoints:
        descriptor = [] #find one descriptor per point
        # desc = sum of neighboring mags
        # ori = orientation[y, x]
        # mag = magnitude[y, x]
        mag_height, mag_width = magnitude.shape

        # 16x16 area, divided into 4x4 areas
        for h in range(-7, 8):
            for k in range(-7, 8):
                 
                # making 4 area 
                hist = np.zeros(4)  #store computed descriptors

                for b in range(x -1, x + 2):
                    for c in range(x -1,x + 2):
                        if b < 0 or c < 0 or b >= mag_width or c >= mag_height:
                            continue

                        sample_ori = orientation[c, b]
                        sample_mag = magnitude[c, b]

                        if math.isnan(mag) or math.isnan(ori):
                            continue

                        # changing the orientation, which rotates the angle
                        rotate_ori = sample_ori - ori
                        i = int(rotate_ori//90)
                        hist[i] += sample_mag

            descriptor.append(hist)

    # flattening descriptor list
    descriptors = []
    for sublist in descriptor:
        for item in sublist:
            descriptors.append(item)

    return descriptors
                
            

def main():
    # read image and preprocessing
    blocks_img = cv.imread('Input_Data/blocks_L-150x150.png')
    blocks_gray = cv.cvtColor(blocks_img, cv.COLOR_BGR2GRAY)


    # Step 1: Scale Space Extrema Detection
    scale_space_img, initial_extrema, octaves = scale_space_extrema(blocks_gray)
    cv.namedWindow('Scale Space Image', cv.WINDOW_AUTOSIZE)
    cv.imshow('Scale Space Image', scale_space_img)

    #Step 2: 
    key_img, new_extrema = key_point_localization(blocks_gray, initial_extrema, octaves)
    cv.namedWindow('KP Localization Image', cv.WINDOW_AUTOSIZE)
    cv.imshow('KP Localization Image', key_img)

    #Step 3:
    ori_image, ori_keys = orientation_assignment(blocks_gray, new_extrema)
    cv.namedWindow('Oriented Image', cv.WINDOW_AUTOSIZE)
    cv.imshow('Oriented Image', ori_image)

    #Step 4:
    mag, ori = find_orientation_magnitude(blocks_gray)
    descriptors = kp_descriptors(ori_keys, mag, ori)

    print(f"Number of created descriptors: {len(descriptors)}")


    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()