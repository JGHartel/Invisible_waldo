# image_game.py
import pandas as pd

class ImageGame:
    def __init__(self, image_path, targets_csv):
        self.image_path = image_path
        self.targets = pd.read_csv(targets_csv)
        self.found_targets = set()

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
