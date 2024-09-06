import os
import cv2

import skimage.color
from skimage import morphology

import numpy as np
import matplotlib.pyplot as plt

def segment(folder_path_load, folder_path_save=None, depict_all=False):
    """
    Segment coins from images in a specified folder, optionally saving the segmented coins.
    We apply edge detection, morphological operations, and circle detection to identify and segment coins in the images. 
    We optionally save the segmented coins to a specified folder if `folder_path_save` argument is passed. 
    Additionally, we generate plots showing intermediate processing steps and segmented coins for each image if `depict_all=True`.
    """
    # create a folder to save segmented coins
    if folder_path_save is not None:
        os.makedirs(folder_path_save, exist_ok=True)

    # path of all images in the folder
    images_path = [folder_path_load+f for f in os.listdir(folder_path_load)]

    # scaling ratio
    ratio = 0.1

    for idx, image_path in enumerate(images_path):
        # read the image
        img = cv2.imread(image_path)

        # convert into RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # detect the background
        type = detect_background(img)

        # convert into grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # detect edges
        if type == 'Neutral':
            img_canny = cv2.Canny(img_gray, 10, 100, 3)
        elif type == 'Hand':
            img_canny = cv2.Canny(img_gray, 50, 200, 3)
        else: 
            img_h, img_s, _ = extract_hsv_channels(img)
            img_s = np.round((img_s*255),0)
            img_binary1 = np.where((img_s < 100) | (img_s >255) , 0, 1)
            img_binary3 = np.where((img_s < 50) | (img_s >80) , 0, 1)
            img_binary1 = img_binary1 | img_binary3
            img_h= np.round((img_h*255),0)
            img_binary2 = np.where((img_h < 10) | (img_h >30) , 0, 1)
            img_canny = img_binary2 & img_binary1
        img_canny = img_canny.astype(np.uint8)*255
        
        # dilation for the images from canny
        if type != 'Noisy':
            img_dilated = cv2.dilate(img_canny, (1000, 1000), iterations=10)
        else: 
            img_dilated = img_canny

        # scale the image
        img_small = cv2.resize(img_dilated, (600, 400))

        # apply morphology
        if type == 'Noisy':
            img_tresh = apply_tresh_morpho_noisy(img_small,ratio)
        else:
            img_tresh = apply_tresh_morpho(img_small, ratio)
        img_uint8 = np.uint8(img_tresh) * 255

        # find the position of the coins
        circles = find_coins(img_uint8)

        if circles is not None:
            # map the coins positions back to the original image
            circles_in_original_image = scale_circles_back(circles)

            # extract the coins from the original image
            extracted_coins = extract_circle_regions(img.copy(), circles_in_original_image)
        else:
            extracted_coins = []

        # example plots
        if idx == 0 or depict_all:
            _, axs = plt.subplots(ncols=4, nrows=1, figsize=(16, 4))
            axs[0].imshow(img)
            axs[0].set_title('original image')
            axs[1].imshow(img_canny)
            axs[1].set_title('extracted edges')
            axs[2].imshow(img_dilated)
            axs[2].set_title('dilated image')
            axs[3].imshow(img_uint8)
            axs[3].set_title('post morphology')
            [axs[i].axis('off') for i in range(4)]
            plt.suptitle(f"e.g. {image_path.split('/')[-1].split('.')[0]} - Detected background: {type}")
            plt.tight_layout()
            plt.show()

            if len(extracted_coins) == 0:
                print('No coins found:')
            else:
                _, axs = plt.subplots(ncols=len(extracted_coins), figsize=(16,3))
                for i, coin in enumerate(extracted_coins):
                    if len(extracted_coins) == 1:
                        axs.imshow(coin)
                        axs.axis('off')
                    else:
                        axs[i].imshow(coin)
                        axs[i].axis('off')
                plt.suptitle(f'Number of coins: {len(extracted_coins)}')
                plt.tight_layout()
                plt.show()

        # get the image name
        image_name = image_path.split('/')[-1].split('.')[0]

        # save all the coins into another folder
        if folder_path_save is not None:
            for i, coin in enumerate(extracted_coins):
                save_path = os.path.join(folder_path_save, f'{image_name}_{i}.JPG')
                cv2.imwrite(save_path, cv2.cvtColor(coin, cv2.COLOR_BGR2RGB))

def extract_circle_regions(image, circles, size=760):
    """
    Extract a square of 760x760 around each coin to ensure to keep the original
    size ratio of each coin.
    """
    extracted_regions = []
    half_size = size // 2
    
    for circle in circles:
        x, y, _ = circle
        
        # Calculate the coordinates of the bounding box centered on (x, y) with fixed size
        x_min = max(0, x - half_size)
        y_min = max(0, y - half_size)
        x_max = min(image.shape[1], x + half_size)
        y_max = min(image.shape[0], y + half_size)
        
        # Ensure the bounding box is exactly `size` x `size`
        if x_max - x_min < size:
            if x_min == 0:
                x_max = size
            else:
                x_min = x_max - size
        if y_max - y_min < size:
            if y_min == 0:
                y_max = size
            else:
                y_min = y_max - size

        # Extract the region
        region = image[y_min:y_max, x_min:x_max, :]
        extracted_regions.append(region)
    
    return extracted_regions

def extract_hsv_channels(img):
    """
    Extract HSV channels from the input image.
    """

    # Get the shape of the input image
    M, N, C = np.shape(img)

    # Define default values for HSV channels
    data_h = np.zeros((M, N))
    data_s = np.zeros((M, N))
    data_v = np.zeros((M, N))

    img_th = skimage.color.rgb2hsv(img)
    data_h = img_th[:, :, 0]
    data_s = img_th[:, :, 1]
    data_v = img_th[:, :, 2]

    return data_h, data_s, data_v

def apply_tresh_morpho(img,ratio):
    """
    Apply morphology methods the images with hand and neutral background
    """    
    d_holes = int(np.round(45000*ratio*ratio,0))
    d_obj = int(np.round(4,0))
    d_closing = int(np.round(300*ratio*ratio,0))
    img_th = img.copy()

    img_th = apply_closing(img_th,d_closing)
    img_th = remove_objects(img_th,d_obj)
    img_th = remove_holes(img_th,d_holes)

    d_obj = int(np.round(20000*ratio*ratio,0))
    img_th = remove_objects(img_th,d_obj) 

    img_morph = img_th
    return img_morph

def apply_tresh_morpho_noisy(img,ratio):
    """
    Apply morphology methods on a noisy image
    """
    d_obj = int(np.round(50000*ratio*ratio,0))
    d_closing = int(np.round(100*ratio*ratio,0))

    img_th = img.copy()
    img_th = apply_closing(img_th,d_closing)
    img_th = remove_holes(img_th,d_obj)

    d_obj = int(np.round(40000*ratio*ratio,0))
    img_th = remove_objects(img_th,d_obj) 

    img_morph = img_th
    return img_morph

def extract_rgb_channels(img):
    """
    Extract RGB channels from the input image.
    """

    # Get the shape of the input image
    M, N, _ = np.shape(img)

    # Define default values for RGB channels
    data_red = np.zeros((M, N))
    data_green = np.zeros((M, N))
    data_blue = np.zeros((M, N))

    data_red = img[:,:,0]
    data_green = img[:,:,1]
    data_blue = img[:,:,2]
    
    return data_red, data_green, data_blue

def apply_closing(img_th, disk_size):
    """
    Apply closing to input mask image using disk shape.
    """

    # Define default value for output image
    img_closing = np.zeros_like(img_th)
    
    # Get the footprint
    footprint = morphology.disk(radius=disk_size)
    
    # Apply closing
    img_closing = morphology.closing(image=img_th, footprint=footprint)

    return img_closing

def scale_circles_back(circles, scale=10):
    """
    Maps the position of the circles back to the original image.
    """    
    scaled_circles = []
    for circle in circles:
        x, y, radius = circle
        x_scaled = int((x * scale))
        y_scaled = int((y * scale))
        radius_scaled = int(radius * scale)
        scaled_circles.append([x_scaled, y_scaled, radius_scaled])
    
    return np.array(scaled_circles)

def filter_overlapping_circles(circles, overlap_threshold=0.3):
    """
    Remove circles that overlap with others by more than 30% of their area.
    """
    # Sort circles by radius in descending order
    circles = circles[circles[:, 2].argsort()[::-1]]
    
    # Initialize result with the largest circle
    result = [circles[0]]
    
    for circle in circles[1:]:
        is_overlapping = False
        for r_circle in result:
            dx = circle[0] - r_circle[0]
            dy = circle[1] - r_circle[1]
            distance = np.sqrt(dx**2 + dy**2)
            overlap = 1 - (distance / (r_circle[2] + circle[2]))
            if overlap > overlap_threshold:
                is_overlapping = True
                break
        if not is_overlapping:
            result.append(circle)
    
    return np.array(result)

def find_coins(image):
    """
    Find circles in the given image using Hough Circles algorithm
    """
    # Use Hough Circles method to detect circles
    minDist = 10
    param1 = 30
    param2 = 18
    minRadius = 10
    maxRadius = 80
    
   
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1,
        minDist,
        param1=param1, 
        param2=param2,
        minRadius=minRadius,
        maxRadius=maxRadius
    )

    if circles is not None:
        # Convert the circle parameters from float to int
        circles = np.round(circles[0, :]).astype("int")

        # Remove overlapping circles
        circles = filter_overlapping_circles(circles)

        return circles
    else:
        print('No coins were found')
        return None

def apply_opening(img_th, disk_size):
    """
    Apply opening to input mask image using disk shape.
    """

    # Define default value for output image
    img_opening = np.zeros_like(img_th)

    # Get the footprint
    footprint =skimage.morphology.disk(disk_size)
    
    # Apply closing
    img_opening = morphology.opening(img_th, footprint)

    return img_opening
    
def remove_holes(img_th, size):
    """
    Remove holes from input image that are smaller than size argument.
    """

    # Define default value for input image
    img_holes = np.zeros_like(img_th)

    # Remove small holes
    img_holes = morphology.remove_small_holes(ar=img_th, area_threshold=size)

    return img_holes

def remove_objects(img_th, size):
    """
    Remove objects from input image that are smaller than size argument.
    """

    # Define default value for input image
    img_obj = np.zeros_like(img_th)

    # Remove small objects
    img_obj = morphology.remove_small_objects(ar=img_th, min_size=size)

    return img_obj

def detect_background(img):
    """
    Detect the background of the given image from the distribution
    of the red and green channels.
    """
    # Get the standard deviation of red and green channels
    data_red, data_green, _ = extract_rgb_channels(img=img)
    standard_red = np.std(data_red)
    standard_green = np.std(data_green)

    # Classify tresholding the std of red and green channels
    if standard_red > 25: 
        return 'Noisy'
    elif standard_green < 20: 
        return 'Neutral'
    else:
        return 'Hand'