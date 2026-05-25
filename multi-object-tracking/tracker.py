
import os

import numpy as np

from detection import detection_cast, draw_detections, extract_detections
from metrics import iou_score


class Tracker:

    """Generate detections and build tracklets."""

    def __init__(self, return_images=True, lookup_tail_size=80, labels=None):
        self.return_images = return_images  # Return images or detections?
        self.frame_index = 0
        self.labels = labels  # Tracker label list
        self.detection_history = []  # Saved detection list
        self.last_detected = {}
        self.tracklet_count = 0  # Counter to enumerate tracklets

        # We will search tracklet at last lookup_tail_size frames
        self.lookup_tail_size = lookup_tail_size

    def new_label(self):
        """Get new unique label."""
        self.tracklet_count += 1
        return self.tracklet_count - 1

    def init_tracklet(self, frame):
        """Get new unique label for every detection at frame and return it."""
        # Write code here
        # Use extract_detections and new_label
        a=extract_detections(frame)
        bob = a.shape[0]
        for i in range(bob):
            b=self.new_label()
            a[i,0]=b
        return a

    @property
    def prev_detections(self):
        """Get detections at last lookup_tail_size frames from detection_history.

        One detection at one id.
        """
        
        # Write code here
        frame_indices = range(max(0, self.frame_index - self.lookup_tail_size), self.frame_index)
        id_map = {}
        # Process frames in reverse to naturally overwrite older entries
        for idx in reversed(list(frame_indices)):
            id_map.update({det[0]: det for det in self.detection_history[idx]})
        return detection_cast([v for _, v in sorted(id_map.items(), key=lambda x: x[1][1])])

    def bind_tracklet(self, detections):
        """Set id at first detection column.

        Find best fit between detections and previous detections.

        detections: numpy int array Cx5 [[label_id, xmin, ymin, xmax, ymax]]
        return: binded detections numpy int array Cx5 [[tracklet_id, xmin, ymin, xmax, ymax]]
        """
        detections = detections.copy()
        prev_detections = self.prev_detections

        detections = detections.copy()
        prev_detections = self.prev_detections
        current_matched = [False] * len(detections)
        prev_matched = [False] * len(prev_detections)
        bob = len(detections)
        
        for i in range(bob):
            best_match = (-1, -1) 
            for j in range(len(prev_detections)):
                if not prev_matched[j]:
                    iou = iou_score(detections[i][1:], prev_detections[j][1:])
                    if iou > best_match[0]:
                        best_match = (iou, j)
            
            if best_match[0] > 0:
                detections[i][0] = prev_detections[best_match[1]][0]
                prev_matched[best_match[1]] = True
                current_matched[i] = True

        
        for i in range(bob):
            if not current_matched[i]:
                detections[i][0] = self.new_label()
        
        return detection_cast(detections)
    
    def save_detections(self, detections):
        """Save last detection frame number for each label."""
        for label in detections[:, 0]:
            self.last_detected[label] = self.frame_index

    def update_frame(self, frame):
        if not self.frame_index:
            # First frame should be processed with init_tracklet function
            detections = self.init_tracklet(frame)
        else:
            # Every Nth frame should be processed with CNN (very slow)
            # First, we extract detections
            detections = extract_detections(frame, labels=self.labels)
            # Then bind them with previous frames
            # Replacing label id to tracker id is performing in bind_tracklet function
            detections = self.bind_tracklet(detections)

        # After call CNN we save frame number for each detection
        self.save_detections(detections)
        # Save detections and frame to the history, increase frame counter
        self.detection_history.append(detections)
        self.frame_index += 1

        # Return image or raw detections
        # Image usefull to visualizing, raw detections to metric
        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():
    from moviepy.editor import VideoFileClip

    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, "data", "test.mp4"))

    tracker = Tracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == "__main__":
    main()
