from pupil_apriltags import Detector  # AprilTag detector from pupil_apriltags
from pupil_labs.realtime_api.simple import Device, discover_one_device
import numpy as np
import cv2
from psychopy import visual, core, event
from marker import AprilMarker
import pandas as pd
from image import ImageGame
import random
import time


tag_detector = Detector()

ADDRESS = "192.168.144.167"
PORT = "8080"

def load_image_game(stimuli_path):
    image = [None] * 12
    for i in range(12):
        image_path = f'{stimuli_path}/image_{i}.png'
        targets_csv = f'{stimuli_path}/targets_{i}.csv'
        image[i] = ImageGame(image_path, targets_csv)
    return image

def setup_device(address=ADDRESS, port=PORT):
    device = Device(address=address, port=port)
    core.wait(1)
    # check connection
    return device  

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    time_per_frame = 1.0 / frame_rate
    return cap, time_per_frame

def calculate_transformation_matrix(detections, marker_info):
    world_points = []
    screen_points = []

    width, height = win.size  # Get the window size for coordinate conversion

    for detection in detections:
        tag_id = detection.tag_id
        if tag_id in marker_info['marker_id'].values:
            marker = marker_info[marker_info['marker_id'] == tag_id]
            marker.dropna() # Drop rows with NaN values

            # Get the corners of the detected AprilTag (OpenCV coordinates)
            for corner in detection.corners:
                world_points.append(np.array(corner))

            # Get the corners from PsychoPy and convert to OpenCV coordinates
            for corner in marker['marker_corners'].values[0]:
                psychopy_x, psychopy_y = corner

                # Convert PsychoPy coordinates to OpenCV coordinates
                cv_x = psychopy_x + (width / 2)
                cv_y = (height / 2) - psychopy_y
                screen_points.append(np.array([cv_x, cv_y]))

    world_points = np.array(world_points).reshape(-1, 2)
    screen_points = np.array(screen_points).reshape(-1, 2)

    if len(world_points) < 4:
        print('Not enough points to calculate the transformation matrix.')
        return None, None

    # Compute the transformation matrix using the world and screen points
    transformation_matrix, _ = cv2.findHomography(world_points, screen_points)
    if transformation_matrix is not None:
        scale_x = np.linalg.norm(transformation_matrix[:, 0])
        scale_y = np.linalg.norm(transformation_matrix[:, 1])
        distance = (scale_x + scale_y) / 2
        return transformation_matrix, distance
    else:
        return None, None


def transform_gaze(gaze, transformation_matrix, win_size):
    gaze_x = gaze[0]
    gaze_y = gaze[1]

    # Apply the transformation matrix
    gaze_point_homogeneous = np.array([gaze_x, gaze_y, 1.0])
    transformed_gaze_point = transformation_matrix @ gaze_point_homogeneous
    transformed_gaze_point /= transformed_gaze_point[2]

    # OpenCV coordinates
    cv_x, cv_y = transformed_gaze_point[:2]

    # Convert back to PsychoPy coordinates for drawing
    width, height = win_size
    psychopy_x = cv_x - (width / 2)
    psychopy_y = (height / 2) - cv_y

    return np.array([psychopy_x, psychopy_y])


def draw_gaze_marker(win, gaze_point, color='red', size=40):
    marker = visual.Circle(win, radius=size, edges=32, lineColor=color, fillColor=color, pos=gaze_point)
    marker.draw()


# Initialize the PsychoPy window
win = visual.Window(units='pix', color='white', screen=2, fullscr=True)

# Define screen size and tag size
width, height = win.size
tag_size = 200

offset = 50 
# Create four AprilMarker instances (one for each corner)s
tag_positions = [
    ((-width +  tag_size) // 2 + offset, (height - tag_size) // 2 - offset),
    ((width - tag_size) // 2 - offset, (height - tag_size) // 2 - offset),
    ((width - tag_size) // 2 - offset, (-height +  tag_size) // 2 + offset),
    ((-width +  tag_size) // 2 + offset, (-height +  tag_size) // 2 + offset),
]

april_markers = []
for i, position in enumerate(tag_positions):
    marker = AprilMarker(marker_id=i, screen_size=(width, height), tag_size=tag_size, position=position)
    marker.create_image_stim(win)
    april_markers.append(marker)

marker_info = pd.DataFrame(columns=['marker_id', 'marker_pos', 'marker_corners'])
for marker in april_markers:
    marker.draw()
    marker_id, marker_pos, marker_corners = marker.get_information()
    marker_info.at[marker_id, 'marker_id'] = marker_id
    marker_info.at[marker_id, 'marker_pos'] = marker_pos
    marker_info.at[marker_id, 'marker_corners'] = marker_corners
    marker_info.dropna()

# Initialize a mask covering the entire image
mask = visual.GratingStim(
    win,
    tex=None,
    mask='circle',
    size=(width * 2, height * 2),
    color='gray',
    opacity=1.0,
    units='pix'
)

def update_mask_position(gaze_point):
    mask.pos = gaze_point

    import time

# Initialize a dictionary to keep track of gaze times on targets
gaze_on_target_start = {}
def check_targets(gaze_point, image, radius=100):
    current_time = time.time()
    for index, target in image.get_targets().iterrows():
        target_id = target['id']
        target_pos = np.array([target['x'], target['y']])
        distance = np.linalg.norm(gaze_point - target_pos)
        if distance < radius:  # Define a radius for vicinity
            if target_id not in gaze_on_target_start:
                gaze_on_target_start[target_id] = current_time
            else:
                elapsed_time = current_time - gaze_on_target_start[target_id]
                if elapsed_time >= 2.0 and not image.is_target_found(target_id):
                    image.mark_target_found(target_id)
        else:
            gaze_on_target_start.pop(target_id, None)

def draw_found_targets(image):
    for target_id in image.found_targets:
        target = image.targets[image.targets['id'] == target_id].iloc[0]
        cross = visual.ShapeStim(
            win,
            vertices=((0, -20), (0, 20), (0,0), (-20, 0), (20, 0)),
            lineWidth=5,
            closeShape=False,
            lineColor='red',
            pos=(target['x'], target['y'])
        )
        cross.draw()

def draw_remaining_targets_count(image):
    count_text = visual.TextStim(
        win,
        text=f"Remaining Targets: {image.remaining_targets()}",
        pos=(width / 2 - 100, height / 2 - 50),  # Adjust position as needed
        color='black',
        units='pix'
    )
    count_text.draw()


device = setup_device()

# Main loop to display the AprilTags
i=0
gaze_transformed = (0, 0)

images = load_image_game('stimuli')

current_image = random.choice(images)

# Main loop to display the game
while not event.getKeys(['escape']):
    win.flip()
    frame, gaze_data = device.receive_matched_scene_video_frame_and_gaze()
    frame = frame.bgr_pixels
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = tag_detector.detect(frame_gray)

    transformation_matrix, distance = calculate_transformation_matrix(detections, marker_info)

    if transformation_matrix is not None:
        gaze_point = transform_gaze(gaze_data['norm_pos'], transformation_matrix, frame.shape, win.size)
        update_mask_position(gaze_point)
        check_targets(gaze_point, current_image)
    else:
        # If no valid transformation, keep the mask centered
        update_mask_position((0, 0))

    # Draw the game image
    current_image.draw()

    # Draw the mask over the image
    mask.draw()

    # Draw found targets
    draw_found_targets(current_image)

    # Draw remaining targets count
    draw_remaining_targets_count(current_image)

    # Flip the window to update the display
    win.flip()


device.close()

# Cleanup
win.close()
core.quit()
