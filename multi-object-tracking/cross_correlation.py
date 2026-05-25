
import os
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import match_template

from detection import detection_cast, draw_detections, extract_detections
from tracker import Tracker


def gaussian(shape, x, y, dx, dy):

    """Return gaussian for tracking.

    shape: [width, height]
    x, y: gaussian center
    dx, dy: std by x and y axes

    return: numpy array (width x height) with gauss function, center (x, y) and std (dx, dy)
    """
    Y, X = np.mgrid[0 : shape[0], 0 : shape[1]]
    return np.exp(-((X - x) ** 2) / dx**2 - (Y - y) ** 2 / dy**2)


class CorrelationTracker(Tracker):

    """Generate detections and building tracklets."""

    def __init__(self, detection_rate=5, **kwargs):

        super().__init__(**kwargs)
        self.detection_rate = detection_rate  # Detection rate
        self.prev_frame = None  # Previous frame (used in cross correlation algorithm)

    def build_tracklet(self, frame):

        """Between CNN execution uses normalized cross-correlation algorithm (match_template)."""
        detections = []
        pad = 20
        # Apply rgb2gray to frame and previous frame
        current_frame_gray = rgb2gray(frame)
        prev_frame_gray = rgb2gray(self.prev_frame) if self.prev_frame is not None else None
    
        if prev_frame_gray is None or len(self.detection_history) == 0:
           self.prev_frame = frame.copy()
           return detection_cast(detections)
        last_bbox = self.detection_history[-1]
        # For every previous detection
        # Use match_template + gaussian to extract detection on current frame
        for label, xmin, ymin, xmax, ymax in last_bbox:
            xmin, xmax = min(xmin, xmax), max(xmin, xmax)
            ymin, ymax = min(ymin, ymax), max(ymin, ymax)
            # Step 0: Extract prev_bbox from prev_frame
            prev_bbox = prev_frame_gray[ymin:ymax, xmin:xmax]
            if prev_bbox.size == 0:
                continue
            # Step 1: Extract new_bbox from current frame with the same coordinates
            height, width = current_frame_gray.shape
            x0 = max(0, xmin - pad)
            y0 = max(0, ymin - pad)
            x1 = min(width, xmax + pad)
            y1 = min(height, ymax + pad)
            search_region = current_frame_gray[y0:y1, x0:x1]

            if search_region.size == 0:
                continue
            if (search_region.shape[0] < prev_bbox.shape[0] or 
                search_region.shape[1] < prev_bbox.shape[1]):
                continue
            # Step 2: Calc match_template between previous and new bbox
            # Use padding
            corr = match_template(search_region, prev_bbox)
            # Step 3: Then multiply matching by gauss function
            # Find argmax(matching * gauss)
            corr_height, corr_width = corr.shape
            y, x = np.mgrid[0:corr_height, 0:corr_width]
            center_y, center_x = corr_height // 2, corr_width // 2
            sigma = pad / 2  
            gauss = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))
            gauss = gauss / gauss.max()  
            corr_weighted = corr * gauss
            ij = np.unravel_index(np.argmax(corr_weighted), corr_weighted.shape)
            dy, dx = ij[0], ij[1]
            # Step 4: Append to detection list
            new_xmin = x0 + dx
            new_ymin = y0 + dy
            new_xmax = new_xmin + (xmax - xmin)
            new_ymax = new_ymin + (ymax - ymin)
            new_xmin = max(0, new_xmin)
            new_ymin = max(0, new_ymin)
            new_xmax = min(width, new_xmax)
            new_ymax = min(height, new_ymax)
            if 0 > new_xmin >= new_xmax or 0 > new_ymin >= new_ymax:
                continue
            
            detections.append((label, new_xmin, new_ymin, new_xmax, new_ymax))
        return detection_cast(detections)

    def update_frame(self, frame):

        if not self.frame_index:
            detections = self.init_tracklet(frame)
            self.save_detections(detections)
        elif self.frame_index % self.detection_rate == 0:
            detections = extract_detections(frame, labels=self.labels)
            detections = self.bind_tracklet(detections)
            self.save_detections(detections)
        else:
            detections = self.build_tracklet(frame)

        self.detection_history.append(detections)
        self.prev_frame = frame
        self.frame_index += 1

        if self.return_images:
            return draw_detections(frame, detections)
        else:
            return detections


def main():

    from moviepy.editor import VideoFileClip

    dirname = os.path.dirname(__file__)
    input_clip = VideoFileClip(os.path.join(dirname, "data", "test.mp4"))

    tracker = CorrelationTracker()
    input_clip.fl_image(tracker.update_frame).preview()


if __name__ == "__main__":

    main()
