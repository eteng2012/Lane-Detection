from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display
import cv2
import os

# Initialize Ikomia Workflow
detector = Workflow()

# Add YOLOP_V2 model to algorithm
model = detector.add_task(name="infer_yolop_v2", auto_connect=True)

# Setting parameters for YOLOP_V2 model
# Setting input size, confidence/min overlap thresholds, and detecting for vehicles and lanes
model.set_parameters({
    "input_size": "640",
    "conf_thres": "0.2",
    "iou_thres": "0.45",
    "object": "True",
    "road_lane": "True"
})

def img_pipeline(image): 

    # Run model on image  
    detector.run_on(image)

    # Retrieve image with bounding boxes on vehicles
    img_detect = model.get_image_with_graphics()
    # Retrieve overlay mask for lanes
    img_lane = model.get_output(0).get_overlay_mask()

    # Resize lane overlay from RGBA to RGB
    img_lane_resized = img_lane[:, :, :3]

    # Overlay lanes onto image with bounding boxes
    overlayed_img = cv2.addWeighted(img_detect, 1, img_lane_resized, 0.8, 0)

    return overlayed_img

from moviepy.editor import VideoFileClip
from IPython.display import HTML

input = 'sample.mp4' # input file location
input_clip = VideoFileClip(input) # makes video object, input file location
output_clip = input_clip.fl_image(img_pipeline) # applies function to each frame
output = 'output.mp4' # output file locatdeion
output_clip.write_videofile(output, audio=False) # writes to new file output