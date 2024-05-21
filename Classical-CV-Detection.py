import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math

# Crops out everythink except bottom triangle by applying a mask
def crop_extra(img, region_to_keep):

    # Create 0 matrix representing blank image, same dimensions
    crop_mask = np.zeros_like(img)

    # Because input image is grayscale, use 255 for white
    safe_area_color = 255
      
    # Fills all locations in mask inside kept region as color=255
    cv2.fillPoly(crop_mask, region_to_keep, safe_area_color)
    
    # Crops out from img anything in that is not 255 in corresponding mask index
    masked_image = cv2.bitwise_and(img, crop_mask)
    return masked_image


# Draws given lines on image
def draw_lines(img, lines, color = [255, 0, 0], thickness = 5):

    # If there are no lines to draw, exit.
    if lines is None:
        return
    
    # Make a copy of the original image
    img = np.copy(img)

    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            img.shape[2]
        ),
        dtype=np.uint8,
    )
    # Loop over all lines and draw them on the new blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
            # Draws straight line on image

    # Merge the image with the lines onto the original.

    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    # img = image to be overlayed, 0.8 = weight/contribution of first image, line_img = image to overlay
    # 1.0 = weight of second image, 0.0 = scalar value added to each pixel (brightness adjustment)

    # Return the modified image.
    return img

def img_pipeline(image):

    # Make region to keep into bottom triangle
    region_to_keep_vertices = [
        (0, image.shape[0]),
        (image.shape[1] / 2, image.shape[0] / 2),
        (image.shape[1], image.shape[0]),
    ]

    # Convert to grayscale for simpler gradient detection
    gray_img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Canny detects strong color function gradient with intensity threshold parameters
    # Threshold parameters to decide how strong gradient counts as edge
    # Two thresholds, one in which below = not edge, one in which above = definitely edge
    # In between, only counts as edge if connected to something above top threshold (area bleeds in)
    canny_img = cv2.Canny(gray_img, 50, 150)

    # Run cropping to get bottom triangle
    cropped_img = crop_extra(
        canny_img,
        np.array([region_to_keep_vertices], np.int32)
    )

    # Now need to turn edge pixels into actual lines
    # Using Hough Transform, will transform edge pixels into different mathematical form
    # Each pixel in image space becomes line/curve in Hough Space
    # In Hough Space, each line represents point in Image Space, each point represents line from Image Space

    # In Image Space, lines defined by y = ax+b, based on two defined (x,y) pairs
    # In Feature Space, axis are a and b, use the line definition and each of the two points (xy pairs) to define line for b
    # b = -xi * a + yi, b = -xj * a + yj
    # Basically, function of b given constants xy and a as the independent variable (rotation around the point)
    # The intersection of the two lines are the correct a and b for the line definition in Image Space

    # Coming back to our CV project, if there are a bunch of points on a line, 
    # then in the feature space those lines should intersect at a point, giving the a/b values for the line in Image Space
    # Algorithm uses row = radius, theta = angle instead of a and b

    # The algorithm will iterate over every pixel row by row, and for each point that is edge, will check row/theta
    # Uses those row/theta to make line, the peak intersections of the lines makes the edges

    lines = cv2.HoughLinesP(
        cropped_img, #image being performed on
        rho = 3, #distance resolution/precision
        theta = np.pi / 60, #angle resolution/precision
        threshold = 160, #min threshold (votes in Hough Space) needed to detect line
        lines = np.array([]), #storage of detected lines
        minLineLength = 40, #Shorter lines discarded
        maxLineGap = 25 #If gap between lines larger, considered separate lines
    )

    # If empty, do not modify image, return
    if (len(lines) == 0):
        return image

    # Group detected lines into left lane and right lane, filter lines not moving toward horizon
    left_x_coord = []
    left_y_coord = []
    right_x_coord = []
    right_y_coord = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if math.fabs(slope) < 0.5: # If slope not > 0.5, not moving toward horizon, not lane
                continue
            if slope <= 0: # Negative slope = line on the left
                left_x_coord.extend([x1, x2])
                left_y_coord.extend([y1, y2])
            else: # Positive slope = line on the right
                right_x_coord.extend([x1, x2])
                right_y_coord.extend([y1, y2])

    # Now need to make a single line representing each line group

    # Set the min and max y values for the line
    min_y = int(0.6 * image.shape[0]) # Horizon cutoff of image
    max_y = image.shape[0] # Bottom of image

    # If empty, do not modify image, return
    if (len(left_x_coord) == 0 or len(right_x_coord) == 0):
        return image

    # poly1d/Polynomial makes polynomial of 1 degree, linear, using given coefficients
    # polyfit returns coefficients of polynomal of given degree that best fits data

    # Calculates polynomial with lines, then draws straight line using min_y/max_y
    from numpy.polynomial import Polynomial

    poly_left = Polynomial(np.flip(np.polyfit(
        left_y_coord,
        left_x_coord,
        deg=1
    )))
    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))
    poly_right = Polynomial(np.flip(np.polyfit(
        right_y_coord,
        right_x_coord,
        deg=1
    )))
    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))

    # Draw lines onto image
    lane_detect_image = draw_lines(
        image, 
        [[
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y],
        ]],
        thickness=5)
    
    return lane_detect_image

from moviepy.editor import VideoFileClip
from IPython.display import HTML

input = 'sample.mp4' # input file location
input_clip = VideoFileClip(input) # makes video object, input file location
output_clip = input_clip.fl_image(img_pipeline) # applies function to each frame
output = 'output.mp4' # output file location
output_clip.write_videofile(output, audio=False) # writes to new file output