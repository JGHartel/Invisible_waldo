# image_game.py
import pandas as pd
import cv2

class ImageGame:
    def __init__(self, image_path, targets_csv):
        self.image_path = image_path
        self.targets = pd.read_csv(targets_csv)
        self.found_targets = set()
        self.image_size = cv2.imread(image_path).shape[:2]
        self.height = self.image_size[0]
        self.width = self.image_size[1]
        self.presentation_size = (1200, 1200)
        self._add_psychopy_coords()

    def get_image(self):
        return self.image_path

    def get_targets(self):
        return self.targets

    def mark_target_found(self, target_id):
        self.found_targets.add(target_id)

    def is_target_found(self, target_id):
        return target_id in self.found_targets

    def remaining_targets(self):
        return len(self.targets) - len(self.found_targets)
    
    def reset(self):
        self.found_targets = set()

    def _add_psychopy_coords(self):
        for index, target in self.targets.iterrows():
            x = target['x']
            y = target['y']

            factor = self.presentation_size[0] / self.width

            #first, convert to presentation size
            x = x * factor
            y = y * factor

            #second, convert to psychopy coordinates
            x = x - self.presentation_size[0] / 2
            y = self.presentation_size[1] / 2 - y

            self.targets.at[index, 'psychopy_x'] = x
            self.targets.at[index, 'psychopy_y'] = y



