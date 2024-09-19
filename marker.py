import os
from PIL import Image
from psychopy import visual
import numpy as np
from pupil_labs.real_time_screen_gaze import marker_generator


class AprilMarker:
    def __init__(self, marker_id, screen_size, tag_size, position=[0, 0], output_dir='april_tags'):
        """
        Initialize an AprilMarker instance.
        
        Parameters:
        - marker_id: The ID of the AprilTag (unique for each marker).
        - screen_size: The screen size (width, height) in pixels.
        - tag_size: Size of the AprilTag in pixels.
        - position: Position (x, y) on the screen where the tag will be displayed.
        - output_dir: Directory where the AprilTag images will be saved.
        """
        self.marker_id = marker_id
        self.screen_size = screen_size
        self.tag_size = tag_size
        self.output_dir = output_dir

        self.corners = np.array([[tag_size/2, tag_size/2], [-tag_size/2, tag_size], [-tag_size/2, -tag_size/2], [tag_size/2, -tag_size/2]]) + np.array(position) 
        
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Generate AprilTag and save it as an image
        self.marker_image_path = self._generate_marker_image()

        # Position: either provided or set to default (center of screen)
        self.position = position if position is not None else (0, 0)

        self.center = np.array(position) + np.array([tag_size, tag_size]) / 2 
        
        # PsychoPy ImageStim object for displaying the tag
        self.image_stim = None

    def _generate_marker_image(self):
        """Generates an AprilTag image and saves it to a file."""
        marker_pixels = marker_generator.generate_marker(marker_id=self.marker_id)
        marker_image = Image.fromarray(marker_pixels)
        marker_image_path = os.path.join(self.output_dir, f'marker_{self.marker_id:02d}.png')
        marker_image.save(marker_image_path)
        return marker_image_path

    def create_image_stim(self, win):
        """
        Creates a PsychoPy ImageStim object to display the AprilTag.
        
        Parameters:
        - win: PsychoPy window where the tag will be displayed.
        
        Returns:
        - image_stim: A PsychoPy ImageStim object to draw the marker.
        """
        self.image_stim = visual.ImageStim(win, image=self.marker_image_path, size=self.tag_size, pos=self.position)
        return self.image_stim

    def draw(self):
        """Draws the AprilTag on the PsychoPy window."""
        if self.image_stim is not None:
            #set autodraw for images
            self.image_stim.autoDraw = True
            self.image_stim.draw()

    def get_information(self):
        """returns a dataframe, where each row has a marker_id string, center position float and corner np.array"""
        return self.marker_id, self.center, self.corners