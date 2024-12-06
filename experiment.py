from pupil_apriltags import Detector  # AprilTag detector from pupil_apriltags
from pupil_labs.realtime_api.simple import Device
import numpy as np
import cv2
from psychopy import visual, core, event, gui
from marker import AprilMarker
import pandas as pd
from image import ImageGame
import random 
import time

tag_detector = Detector()

ADDRESS = "192.168.137.138"
PORT = "8080"
LANGUAGE = "en"

def load_image_game(stimuli_path):
    image = [None] * 12
    for i in range(1,13):
        image_path = f'{stimuli_path}/image_{i}.jpg'
        targets_csv = f'{stimuli_path}/targets_{i}.csv'
        image[i-1] = ImageGame(image_path, targets_csv)
    return image

def setup_device(address=ADDRESS, port=PORT):
    device = Device(address=address, port=port)
    # check connection
    return device  

def wait_for_keypress():
    while not event.getKeys():
        core.wait(0.1)  # Wait for 100ms before checking again

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


# Initialize the PsychoPy window
win = visual.Window(units='pix', color='white', fullscr=True, allowStencil=True, screen=1) 

# Define screen size and tag size
width, height = win.size
tag_size = 300

offset = 20 
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
    marker_id, marker_pos, marker_corners = marker.get_information()
    marker_info.at[marker_id, 'marker_id'] = marker_id
    marker_info.at[marker_id, 'marker_pos'] = marker_pos
    marker_info.at[marker_id, 'marker_corners'] = marker_corners
    marker_info.dropna()

def draw_markers():
    for marker in april_markers:
        marker.draw()

# Function to generate a circular aperture for the gaze
def create_gaze_aperture(win, radius=200):
    # Create a circular mask with a transparent aperture
    aperture = visual.Aperture(win, size=2*radius, shape='circle')
    aperture.enabled = True  # Initially enable the aperture
    return aperture

aperture = create_gaze_aperture(win)    

# Function to update the aperture position to match the gaze
def update_aperture_position(gaze_point):
    aperture.pos = gaze_point


# Initialize a dictionary to keep track of gaze times on targets
gaze_on_target_start = {}

def check_targets(gaze_point, image, radius=200):
    current_time = time.time()
    for index, target in image.targets.iterrows():
        target_id = target['id']
        target_pos = np.array([target['psychopy_x'], target['psychopy_y']])
        distance = np.linalg.norm(gaze_point - target_pos)
        if distance < radius:  # Define a radius for vicinity
            if target_id not in gaze_on_target_start:
                gaze_on_target_start[target_id] = current_time
            else:
                elapsed_time = current_time - gaze_on_target_start[target_id]
                if elapsed_time >= 1.0 and not image.is_target_found(target_id):
                    image.mark_target_found(target_id)

        else:
            gaze_on_target_start.pop(target_id, None)

crosses = {}

def draw_found_targets(win, image):
    # Iterate over found targets
    for target_id in image.found_targets:
        if target_id not in crosses:
            # Retrieve target position from the image data
            target = image.targets[image.targets['id'] == target_id].iloc[0]
            
            # Create a cross marker at the target's position
            cross = visual.ShapeStim(
                win,
                vertices=((0, -20), (0, 20), (0, 0), (-20, 0), (20, 0)),
                lineWidth=5,
                closeShape=False,
                lineColor='red',
                pos=(target['psychopy_x'], target['psychopy_y'])
            )
            
            # Add the cross to the dictionary of drawn crosses, keyed by target ID
            crosses[target_id] = cross

    # Ensure all previously drawn crosses remain on screen
    for cross in crosses.values():
        cross.draw()

    return crosses

def reset_crosses(crosses):
    crosses = {}
    return crosses

def draw_remaining_targets_count(image):
    #check if count_text is defined
    if 'count_text' in locals():
        # delete the object
        del count_text

    if LANGUAGE == "en":
        count_text = visual.TextStim(
            win,
            text=f"Remaining Targets: {image.remaining_targets()}",
            pos=(0, height / 2 - 50),  # Adjust position as needed
            height=50,
            color='red',
            units='pix'
        )
    elif LANGUAGE == "de":
        count_text = visual.TextStim(
            win,
            text=f"Verbleibende Ziele: {image.remaining_targets()}",
            pos=(0, height / 2 - 50),  # Adjust position as needed
            height=50,
            color='red',
            units='pix'
        )

    count_text.draw()



# Main loop to display the AprilTags
i=0
gaze_transformed = (0, 0)

images = load_image_game('stimuli')

current_image = random.choice(images)

game_image = visual.ImageStim(win, image=current_image.get_image(), units='pix', pos=(0, 0), size=(1200, 1200))

aperture.enabled = False

gaze_point = np.zeros(2)
prev_gaze_point = np.zeros((3, 2))


participant_info = {'Name': ''}
dlg = gui.DlgFromDict(participant_info, title='Participant Information')
if not dlg.OK:
    core.quit()
participant_name = participant_info['Name']


device = setup_device()
core.wait(1)  # Wait for 1 second to establish connection

# Draw the markers
draw_markers()

if LANGUAGE == "en":
    intro_text = visual.TextStim(
        win,
        text='Welcome to the experiment!\n\nYour task is to find all clovers on screen.\n\nPress any key to start.',
        pos=(0, 0),
        height=30,
        color='black',
        units='pix'
    )
elif LANGUAGE == "de":
# Introduction text
    intro_text = visual.TextStim(
        win,
        text='Willkommen zum Experiment!\n\nIhre Aufgabe ist es, alle Kleeblätter auf dem Bildschirm zu finden.\n\nDrücken Sie eine beliebige Taste, um zu beginnen.',
        pos=(0, 0),
        height=30,
        color='black',
        units='pix'
    )

# Display the introduction text
intro_text.draw()
win.flip()
wait_for_keypress()

# Main loop to display the game
results = []
for trial in range(1, 4):
    current_image = random.choice(images)
    game_image = visual.ImageStim(win, image=current_image.get_image(), units='pix', pos=(0, 0), size=(1200, 1200))

    current_image.reset() 
    trial_start_time = time.time()
    
    while not event.getKeys(['escape']):
        frame, gaze_data = device.receive_matched_scene_video_frame_and_gaze()

        frame = frame.bgr_pixels
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = tag_detector.detect(frame_gray)

        
        transformation_matrix, distance = calculate_transformation_matrix(detections, marker_info)

        if transformation_matrix is not None:
            gaze_point = transform_gaze(gaze_data, transformation_matrix, win.size)
            
            update_aperture_position(gaze_point)
            check_targets(gaze_point, current_image)
        elif transformation_matrix is None:
            gaze_point = (0, 0)

            update_aperture_position(gaze_point)
            check_targets(gaze_point, current_image)

        aperture.enabled = True
        # Draw the game image
        game_image.draw()
        # Draw the mask over the image
        aperture.enabled = False
        draw_markers()
        # Draw found targets
        crosses = draw_found_targets(win, current_image) 

        # Draw remaining targets count
        draw_remaining_targets_count(current_image)
        win.flip()

        # Break the loop if all targets are found
        if current_image.remaining_targets() == 0:
            break

    crosses = reset_crosses(crosses)

    # Record the end time of the trial
    trial_end_time = time.time()
    trial_duration = trial_end_time - trial_start_time

    # Record the number of targets found
    targets_found = len(current_image.found_targets)
    total_targets = len(current_image.targets)

    # Save the trial results
    trial_result = {
        'Name': participant_name,
        'Trial': trial,
        'Duration': trial_duration,
        'TargetsFound': targets_found,
        'TotalTargets': total_targets
    }
    results.append(trial_result)

    # Optional: Display a short break or instruction between trials
    if trial < 3:
        if LANGUAGE == "en":
            break_text = visual.TextStim(
                win,
                text='Get ready for the next round!\n\nPress any key to continue.',
                pos=(0, 0),
                height=30,
                color='black',
                units='pix'
            )

        elif LANGUAGE == "de":
            break_text = visual.TextStim(
                win,
                text='Machen Sie sich bereit für die nächste Runde!\n\nDrücken Sie eine beliebige Taste, um fortzufahren.',
                pos=(0, 0),
                height=30,
                color='black',
                units='pix'
            )

        break_text.draw()
        win.flip()
        wait_for_keypress()

results_df = pd.DataFrame(results)

# sum the time one participant took to find all targets
total_time = results_df['Duration'].sum()
total_targets_found = results_df['TargetsFound'].sum()
time_per_target = total_time / total_targets_found 

final_results = {
    'Name': participant_name,
    'TotalTime': total_time,
    'TotalTargetsFound': total_targets_found,
    'TimePerTarget': time_per_target
}


import csv
import os

results_file = 'leaderboard.csv'

# Check if the file exists
file_exists = os.path.isfile(results_file)

with open(results_file, 'a', newline='') as csvfile:
    fieldnames = ['Name', 'TotalTime', 'TotalTargetsFound', 'TimePerTarget']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # If the file doesn't exist, write the header
    if not file_exists:
        writer.writeheader()

    writer.writerow(final_results)

# Display the final results
if LANGUAGE == "en":
    final_text = visual.TextStim(
        win,
        text=f'Congratulations, {participant_name}!\n\nYou found {total_targets_found} targets in {total_time:.2f} seconds.\n\nAverage time per target: {time_per_target:.2f} seconds.',
        pos=(0, 0),
        height=30,
        color='black',
        units='pix'
    )
elif LANGUAGE == "de":
    final_text = visual.TextStim(
        win,
        text=f'Herzlichen Glückwunsch, {participant_name}!\n\nSie haben {total_targets_found} Ziele in {total_time:.2f} Sekunden gefunden.\n\nDurchschnittliche Zeit pro Ziel: {time_per_target:.2f} Sekunden.',
        pos=(0, 0),
        height=30,
        color='black',
        units='pix'
    )

final_text.draw()
win.flip()
wait_for_keypress()


# display leaderboard
leaderboard = pd.read_csv('leaderboard.csv')
leaderboard = leaderboard.sort_values(by='TimePerTarget', ascending=True)
# reset index
leaderboard.reset_index(drop=True, inplace=True)

if LANGUAGE == "en":
    leaderboard_text = 'Leaderboard\n\n'
    for index, row in leaderboard.iterrows():
        if index < 5:
            leaderboard_text += f"{index + 1}. {row['Name']}: {row['TimePerTarget']:.2f} seconds\n"

    leaderboard_text += '\nPress any key to exit.'
elif LANGUAGE == "de":
    leaderboard_text = 'Bestenliste\n\n'
    for index, row in leaderboard.iterrows():
        if index < 5:
            leaderboard_text += f"{index + 1}. {row['Name']}: {row[ 'TimePerTarget']:.2f} Sekunden\n"

    leaderboard_text += '\nDrücken Sie eine beliebige Taste, um zu beenden.'

leaderboard_text_stim = visual.TextStim(
    win,
    text=leaderboard_text,
    pos=(0, 0),
    height=50,
    color='black',
    units='pix'
)

leaderboard_text_stim.draw()
win.flip()
wait_for_keypress()

device.close()

# Cleanup
win.close()
core.quit()
