
import numpy as np
def iou_score(bbox1, bbox2):
    """Jaccard index or Intersection over Union.

    https://en.wikipedia.org/wiki/Jaccard_index

    bbox: [xmin, ymin, xmax, ymax]
    """
    assert len(bbox1) == 4
    assert len(bbox2) == 4

    # Write code here

    first=np.array(bbox1)
    second=np.array(bbox2)
    a=np.min([first[2:],second[2:]],axis=0)
    b=np.max([first[:2],second[:2]],axis=0)
    mutual=np.prod(np.max([a-b,np.zeros(2)],axis=0))
    merge=np.prod(first[2:]-first[:2])+np.prod(second[2:]-second[:2])-mutual
    # print(mutual,merge)
    return mutual/merge 


def motp(obj, hyp, threshold=0.5):
    """Calculate MOTP

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0

    matches={}
    matched_pairs = []  # matches between object IDs and hypothesis IDs

    # For every frame
    for real_obj, hyp_obj in zip(obj, hyp):
        # Write code here

        # Step 1: Convert frame detections to dict with IDs as keys
        obj = {o[0]: o[1:] for o in real_obj}  
        hyp = {h[0]: h[1:] for h in hyp_obj}  
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        for obj_id, obj_bbox in obj.items():
            for hyp_id, hyp_bbox in hyp.items():
                iou = iou_score(obj_bbox, hyp_bbox)
                if  iou > threshold:
                    dist_sum += iou
                    match_count += 1
                    matched_pairs.append((obj_id, hyp_id))
            
        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold

        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections

        # Step 5: Update matches with current matched IDs
        

    # Step 6: Calculate MOTP
    if match_count == 0:
        return 0  # No matches, return 0 as MOTP

    MOTP = dist_sum / match_count
    return MOTP



def motp_mota(obj, hyp, threshold=0.5):
    """Calculate MOTP/MOTA

    obj: list
        Ground truth frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    hyp: list
        Hypothetical frame detections.
        detections: numpy int array Cx5 [[id, xmin, ymin, xmax, ymax]]

    threshold: IOU threshold
    """

    dist_sum = 0  # a sum of IOU distances between matched objects and hypotheses
    match_count = 0
    missed_count = 0
    false_positive = 0
    mismatch_error = 0

    matches = {}  # matches between object IDs and hypothesis IDs
    c = sum(len(frame) for frame in obj)
    # For every frame
    for real_obj, hyp_obj in zip(obj, hyp):
        # Step 1: Convert frame detections to dict with IDs as keys
        detections = {det[0]: det[1:5] for det in real_obj}
        hyp_detections = {det[0]: det[1:5] for det in hyp_obj}
        # Step 2: Iterate over all previous matches
        # If object is still visible, hypothesis still exists
        # and IOU distance > threshold - we've got a match
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        existing_matches = list(matches.items())
        for obj_id, hyp_id in existing_matches:
            if obj_id in detections and hyp_id in hyp_detections:
                iou = iou_score(detections[obj_id], hyp_detections[hyp_id])
                if iou > threshold:
                    dist_sum += iou
                    match_count += 1
                    del detections[obj_id]
                    del hyp_detections[hyp_id]

        # Step 3: Calculate pairwise detection IOU between remaining frame detections
        # Save IDs with IOU > threshold
        ious = []
        for obj_id in detections:
            for hyp_id in hyp_detections:
                iou = iou_score(detections[obj_id], hyp_detections[hyp_id])
                if iou > threshold:
                    ious.append((iou, obj_id, hyp_id))
        # Step 4: Iterate over sorted pairwise IOU
        # Update the sum of IoU distances and match count
        # Delete matched detections from frame detections
        ious.sort(key=lambda x: x[0])

        for iou, obj_id, hyp_id in ious:
            if obj_id not in detections or hyp_id not in hyp_detections:
                continue
        # Step 5: If matched IDs contradict previous matched IDs - increase mismatch error
            existing_obj_ids = [k for k, v in matches.items() if v == hyp_id]

            if obj_id in matches and matches[obj_id] != hyp_id:
                mismatch_error += 1
        # Step 6: Update matches with current matched IDs
            matches[obj_id] = hyp_id
            dist_sum += iou
            match_count += 1
            del detections[obj_id]
            del hyp_detections[hyp_id]

        # Step 7: Errors
        # All remaining hypotheses are considered false positives
        # All remaining objects are considered misses
        missed_count += len(detections)
        false_positive += len(hyp_detections)

    # Step 8: Calculate MOTP and MOTA
    MOTP = dist_sum / match_count if match_count > 0 else 0.0
    MOTA = 1.0 - (missed_count + false_positive + mismatch_error) / c if c > 0 else 0.0

    return MOTP, MOTA