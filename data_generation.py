import random
import numpy as np
import math
import os
from PIL import Image, ImageDraw
import tqdm

def generate_xf_image(label, length_case, fin_case, image_size=256):
    """
    Generate a Cross Fin (XF) image of size image_size**2

    In these images:
      - We have a 'top line' and a 'bottom line', each drawn horizontally with their centers at the same x-coordinate.
      - We need two labels: 0 (two lines are equal in length) and 1 (one of them is longer).
        - In 1, we have two length_case: "LONG" (top line is longer) and "SHORT" (top line is shorter)
        - In 0, both lines are equal in length so we name length_case "EQUAL".
      - Each line has 'cross fins' at both endpoints: two angled lines forming an 'X'.
        - Fin angle, fin length, and vertical position are randomized for each line in the image.
            - Let's create two fin_case: "SAME_CONFIG" (both lines have the same fin angle and length) and "DIFF_CONFIG" (each line has different fin angles and lengths).

    Returns
    -------
    (img, label, length_case, fin_case) : (PIL.Image, int, str, str)
    """

    # random vertical positions 
    top_y = random.randint(image_size // 5, image_size // 5 * 2)
    bottom_y = random.randint(image_size // 5 * 3, image_size // 5 * 4)

    # shaft lengths based on the label
    min_length = 45
    max_length = int(image_size*0.8)  

    if label == 1: 
        if length_case == "LONG":
            top_length = random.randint(min_length, max_length)
            delta = random.randint(2, 62)
            bottom_length = max(min_length, top_length - delta)
        elif length_case == "SHORT":
            bottom_length = random.randint(min_length, max_length)
            delta = random.randint(2, 62)
            top_length = max(min_length, bottom_length - delta)

    else: 
        length_case = "EQUAL"
        top_length = random.randint(min_length, max_length)
        bottom_length = top_length

    # horizontal positions (coordinates of the left points of the shafts) so that the lines are centered 
    top_x = image_size // 2 - top_length // 2
    bottom_x = image_size // 2 - bottom_length // 2

    # randomize fin lengths and angles for top and bottom lines
    #    Fin angles: 10-90 degrees; fin length: 15-40
    top_fin_length = random.randint(10, min(35, top_length // 3))
    top_fin_angle_deg = random.randint(15, 75)

    if fin_case == "SAME_CONFIG":
        bottom_fin_length = top_fin_length
        bottom_fin_angle_deg = top_fin_angle_deg
    else:
        bottom_fin_length = random.randint(10, min(35, bottom_length // 3))
        bottom_fin_angle_deg = random.randint(15, 75)
    
    # deg to rad
    top_fin_angle_rad = np.radians(top_fin_angle_deg)
    bottom_fin_angle_rad = np.radians(bottom_fin_angle_deg)

    # create image
    img = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(img)
    line_width = 2  

    def draw_cross_fins(draw, center, fin_length, angle_rad):
        """
        Draw an 'X' shape that intersects the horizontal line.
        angle_rad is the angle from the VERTICAL, so 0 means 'straight up/down'.
        """
        x0, y0 = center

        # from endpoint up-left
        xA = x0 - fin_length * math.sin(angle_rad)
        yA = y0 - fin_length * math.cos(angle_rad)
        draw.line([center, (xA, yA)], fill="black", width=2)

        # from endpoint up-right
        xB = x0 + fin_length * math.sin(angle_rad)
        yB = y0 - fin_length * math.cos(angle_rad)
        draw.line([center, (xB, yB)], fill="black", width=2)

        # from endpoint down-left
        xC = x0 - fin_length * math.sin(angle_rad)
        yC = y0 + fin_length * math.cos(angle_rad)
        draw.line([center, (xC, yC)], fill="black", width=2)

        # from endpoint down-right
        xD = x0 + fin_length * math.sin(angle_rad)
        yD = y0 + fin_length * math.cos(angle_rad)
        draw.line([center, (xD, yD)], fill="black", width=2)

    # draw the two horizontal shafts
    top_line_left = (top_x, top_y)
    top_line_right = (top_x + top_length, top_y)
    bottom_line_left = (bottom_x, bottom_y)
    bottom_line_right = (bottom_x + bottom_length, bottom_y)

    draw.line([top_line_left, top_line_right], fill="black", width=line_width)
    draw.line([bottom_line_left, bottom_line_right], fill="black", width=line_width)

    # draw cross fins at each end of the top line
    draw_cross_fins(draw, top_line_left, top_fin_length, top_fin_angle_rad)
    draw_cross_fins(draw, top_line_right, top_fin_length, top_fin_angle_rad)

    # draw cross fins at each end of the bottom line
    draw_cross_fins(draw, bottom_line_left, bottom_fin_length, bottom_fin_angle_rad)
    draw_cross_fins(draw, bottom_line_right, bottom_fin_length, bottom_fin_angle_rad)

    params = {
        "label": label,
        "length_case": length_case,
        "fin_case": fin_case,
        "top_length": top_length,
        "bottom_length": bottom_length,
        "top_fin_length": top_fin_length,
        "top_fin_angle_deg": top_fin_angle_deg,
        "bottom_fin_length": bottom_fin_length,
        "bottom_fin_angle_deg": bottom_fin_angle_deg,
        "top_y": top_y,
        "bottom_y": bottom_y,
    }

    return img, params


def generate_xf_dataset(num_images=2000, case_ratio=(1,1,1,1,1,2), image_size=256):
    """
    Generate a dataset of Cross Fin (XF) images.
    
    Parameters
    ----------
    num_images : int
        Number of images to generate.
    image_size : int
        Size of each square image
    case_ratio : tuple of int
        Ratio of the different cases to be generated.
        (0, 1, LONG, SHORT, SAME_CONFIG, DIFF_CONFIG)
    
    Returns
    -------
    (images, params) : (list of PIL.Image, list of int)
        images: list of generated XF images
        params: list of dictionaries with parameters for each image
    """
    images = []
    configs = []
    for _ in range(num_images):
        lbl = random.randint(0, 1)
        fin_case = random.choices(["SAME_CONFIG", "DIFF_CONFIG"], weights=case_ratio[4:6])[0]
        if lbl == 1:
            length_case = random.choices(["LONG", "SHORT"], weights=case_ratio[2:4])[0]
        else:
            length_case = "EQUAL"
        
        # Generate the image
        img, params = generate_xf_image(lbl, length_case, fin_case, image_size=image_size)
        images.append(img)
        configs.append(params)

    return images, configs

def generate_muller_lyer_image(top_dir_case, bottom_dir_case, fin_case, image_size=256):
    """
    Generates a Müller-Lyer illusion image.
   
    In these images:
        - We have a 'top line' and a 'bottom line', each drawn horizontally with their centers at the same x-coordinate.
        - Both lines are equal in length, so they have the same label 1. 
        - Each line has 'arrowheads' at both endpoints: >---< for inward and <--> for outward.
            - The angle and length of the arrowheads is randomized for each line in the image.
                - There are two top_dir_case: "LONG" (top line has inward arrowheads) and "SHORT" (top line has outward arrowheads).
                  - In both top_dir_case for the top line, there are two bottom_dir_cases: "SAME_DIR" (the bottom line has same arrow direction with the top one) and "DIFF_DIR" (they have opposite arrow directions).
                - There are two fin_case for the arrowheads: "SAME_CONFIG" (both lines have the same fin angle and length) and "DIFF_CONFIG" (each line has different fin angles and lengths).
            - The vertical position between the two lines is also randomized.

    Returns:

    """
    # random vertical positions 
    top_y = random.randint(image_size // 5, image_size // 5 * 2)
    bottom_y = random.randint(image_size // 5 * 3, image_size // 5 * 4)

    # shaft lengths
    min_length = 45
    max_length = int(image_size*0.8)
    shaft_length = random.randint(min_length, max_length)

    top_length = shaft_length
    bottom_length = shaft_length

    # horizontal positions (coordinates of the left points of the shafts) so that the lines are centered
    top_x = image_size // 2 - top_length // 2
    bottom_x = image_size // 2 - bottom_length // 2

    # randomize fin lengths and angles for top and bottom lines
    #    Fin angles: 10-90 degrees; fin length: 15-40
    top_fin_length = random.randint(15, min(35, top_length // 3))
    top_fin_angle_deg = random.randint(15, 75)

    if fin_case == "SAME_CONFIG":
        bottom_fin_length = top_fin_length
        bottom_fin_angle_deg = top_fin_angle_deg
    else:
        bottom_fin_length = random.randint(15, min(35, bottom_length // 3))
        bottom_fin_angle_deg = random.randint(15, 75)
    
    # deg to rad
    top_fin_angle_rad = np.radians(top_fin_angle_deg)
    bottom_fin_angle_rad = np.radians(bottom_fin_angle_deg)

    img = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(img)
    line_width = 2  

    def draw_arrowhead(draw, center, fin_length, angle_rad, direction):
        """
        Draw an arrowhead at the end of a horizontal line segment.
        angle_rad is the angle from the VERTICAL, so 0 means 'straight up/down'.
        """
        x0, y0 = center
        
        if direction == "right": # arrowhead points to the right
            # from endpoint up-left
            xA = x0 - fin_length * math.sin(angle_rad)
            yA = y0 - fin_length * math.cos(angle_rad)
            draw.line([center, (xA, yA)], fill="black", width=2)

            # from endpoint down-left
            xB = x0 - fin_length * math.sin(angle_rad)
            yB = y0 + fin_length * math.cos(angle_rad)
            draw.line([center, (xB, yB)], fill="black", width=2)

        if direction == "left": # arrowhead points to the left
            # from endpoint up-right
            xA = x0 + fin_length * math.sin(angle_rad)
            yA = y0 - fin_length * math.cos(angle_rad)
            draw.line([center, (xA, yA)], fill="black", width=2)

            # from endpoint down-right
            xB = x0 + fin_length * math.sin(angle_rad)
            yB = y0 + fin_length * math.cos(angle_rad)
            draw.line([center, (xB, yB)], fill="black", width=2)

    # draw the two horizontal shafts
    top_line_left = (top_x, top_y)
    top_line_right = (top_x + top_length, top_y)
    bottom_line_left = (bottom_x, bottom_y)
    bottom_line_right = (bottom_x + bottom_length, bottom_y)

    draw.line([top_line_left, top_line_right], fill="black", width=line_width)
    draw.line([bottom_line_left, bottom_line_right], fill="black", width=line_width)

    # draw the arrowheads at each end of the top line
    if top_dir_case == "LONG": # top line has inward arrowheads, so left arrow points to the right and right arrow points to the left
        draw_arrowhead(draw, top_line_left, top_fin_length, top_fin_angle_rad, "right")
        draw_arrowhead(draw, top_line_right, top_fin_length, top_fin_angle_rad, "left")

        if bottom_dir_case == "SAME_DIR":
            draw_arrowhead(draw, bottom_line_left, bottom_fin_length, bottom_fin_angle_rad, "right")
            draw_arrowhead(draw, bottom_line_right, bottom_fin_length, bottom_fin_angle_rad, "left")
        elif bottom_dir_case == "DIFF_DIR":
            draw_arrowhead(draw, bottom_line_left, bottom_fin_length, bottom_fin_angle_rad, "left")
            draw_arrowhead(draw, bottom_line_right, bottom_fin_length, bottom_fin_angle_rad, "right")

    elif top_dir_case == "SHORT": # top line has outward arrowheads, so left arrow points to the left and right arrow points to the right
        draw_arrowhead(draw, top_line_left, top_fin_length, top_fin_angle_rad, "left")
        draw_arrowhead(draw, top_line_right, top_fin_length, top_fin_angle_rad, "right")

        if bottom_dir_case == "SAME_DIR":
            draw_arrowhead(draw, bottom_line_left, bottom_fin_length, bottom_fin_angle_rad, "left")
            draw_arrowhead(draw, bottom_line_right, bottom_fin_length, bottom_fin_angle_rad, "right")
        elif bottom_dir_case == "DIFF_DIR":
            draw_arrowhead(draw, bottom_line_left, bottom_fin_length, bottom_fin_angle_rad, "right")
            draw_arrowhead(draw, bottom_line_right, bottom_fin_length, bottom_fin_angle_rad, "left")

    # Package parameters for return
    params = {
        "label": 0,  # All images are labeled 0 (equal length)
        "top_dir_case": top_dir_case,
        "bottom_dir_case": bottom_dir_case,
        "fin_case": fin_case,
        "shaft_length": shaft_length,
        "top_fin_length": top_fin_length,
        "top_fin_angle_deg": top_fin_angle_deg,
        "bottom_fin_length": bottom_fin_length,
        "bottom_fin_angle_deg": bottom_fin_angle_deg,
        "top_y": top_y,
        "bottom_y": bottom_y,
    }

    return img, params

def generate_muller_lyer_dataset(num_images=2000, case_ratio=(1,1,1,1,1,2), image_size=256):
    """
    Generate a dataset of Müller-Lyer illusion images.
    
    Parameters
    ----------
    num_images : int
        Number of images to generate.
    image_size : int
        Size of each square image
    case_ratio : tuple of int
        Ratio of the different cases to be generated.
        (LONG, SHORT, SAME_DIR, DIFF_DIR, SAME_CONFIG, DIFF_CONFIG)
    
    Returns
    -------
    (images, params) : (list of PIL.Image, list of int)
        images: list of generated Müller-Lyer images
        params: list of dictionaries with parameters for each image
    """
    images = []
    configs = []
    for _ in range(num_images):
        top_dir_case = random.choices(["LONG", "SHORT"], weights=case_ratio[2:4])[0]
        bottom_dir_case = random.choices(["SAME_DIR", "DIFF_DIR"], weights=case_ratio[4:6])[0]
        fin_case = random.choices(["SAME_CONFIG", "DIFF_CONFIG"], weights=case_ratio[4:6])[0]

        # Generate the image
        img, params = generate_muller_lyer_image(top_dir_case, bottom_dir_case, fin_case, image_size=image_size)
        images.append(img)
        configs.append(params)

    return images, configs

def generate_and_save(type, num_images=2000, case_ratio=(1,1,1,1,1,2), image_size=256):
    """
    Generate and save a dataset
    
    Parameters
    ----------
    type : str
        "train" or "test"
    num_images : int
    case_ratio : tuple of int
    image_size : int
    """
    if type == "train":
        imgs, configs = generate_xf_dataset(num_images=num_images, case_ratio=case_ratio, image_size=image_size)
        if not os.path.exists("training_data"):
            os.makedirs("training_data")
        for i, (img, config) in enumerate(zip(imgs, configs)):
            img.save(f"training_data/xf_{i}_{config['label']}_{config['length_case']}_{config['fin_case']}_{config['top_length']}_{config['bottom_length']}_{config['top_fin_length']}_{config['top_fin_angle_deg']}_{config['bottom_fin_length']}_{config['bottom_fin_angle_deg']}_{config['top_y']}_{config['bottom_y']}.png")

    elif type == "test":
        imgs, configs = generate_muller_lyer_dataset(num_images=num_images, case_ratio=case_ratio, image_size=image_size)
        if not os.path.exists("test_data"):
            os.makedirs("test_data")
        for i, (img, config) in enumerate(zip(imgs, configs)):
            img.save(f"test_data/ml_{i}_{config['label']}_{config['top_dir_case']}_{config['bottom_dir_case']}_{config['fin_case']}_{config['shaft_length']}_{config['top_fin_length']}_{config['top_fin_angle_deg']}_{config['bottom_fin_length']}_{config['bottom_fin_angle_deg']}_{config['top_y']}_{config['bottom_y']}.png")
