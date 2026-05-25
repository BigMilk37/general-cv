import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
# ============================== 1 Classifier model ============================
def get_cls_model(input_shape=(1, 40, 100)):
    """
    :param input_shape: tuple (n_rows, n_cols, n_channels)
            input shape of image for classification
    :return: nn model for classification
    """
    classification_model = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.BatchNorm2d(16),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.BatchNorm2d(32),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.BatchNorm2d(128),
        nn.Dropout(0.2),
        nn.ReLU(),
        nn.MaxPool2d(2),
        
        nn.Flatten(),
        nn.Linear(128 * 2 * 6, 2) 
    )
    return classification_model


def fit_cls_model(X, y, fast_train=True):
    """
    :param X: 4-dim tensor with training images
    :param y: 1-dim tensor with labels for training
    :return: trained nn model
    """

    # fast_train = False
    epochs = 1 if fast_train else 40
    batch_size = 4 if fast_train else 32

    model = get_cls_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    
    for i in range(epochs):
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    if not fast_train:
        torch.save(model.state_dict(), "classifier_model.pt")
    return model


# ============================ 2 Classifier -> FCN =============================
def get_detection_model(cls_model):
    """
    :param cls_model: trained cls model
    :return: fully convolutional nn model with weights initialized from cls
             model
    """

    detection_layers = []
    layers = list(cls_model.children())
    for layer in layers:
        if isinstance(layer, nn.Flatten):
            break
        detection_layers.append(layer)
    
    remaining_layers = layers[len(detection_layers):]
    linear_layer = None
    for layer in remaining_layers:
        if isinstance(layer, nn.Linear):
            linear_layer = layer
            break
    
    in_channels = 128
    out_features = linear_layer.out_features
    kernel_size = (2, 6)
    conv_layer = nn.Conv2d(in_channels, out_features, kernel_size=kernel_size)
    
    linear_weight = linear_layer.weight.view(out_features, in_channels, kernel_size[0], kernel_size[1])
    conv_layer.weight.data = linear_weight
    conv_layer.bias.data = linear_layer.bias.data
    
    detection_layers.append(conv_layer)
    detection_model = nn.Sequential(*detection_layers)
    return detection_model



# ============================ 3 Simple detector ===============================
def get_detections(detection_model, dictionary_of_images):
    """
    :param detection_model: trained fully convolutional detector model
    :param dictionary_of_images: dictionary of images in format {filename: ndarray}
    :return: detections in format {filename: detections}. detections is a N x 5
        array, where N is number of detections. Each detection is described
        using 5 numbers: [row, col, n_rows, n_cols, confidence].
    """
    detections = {}
    effective_stride = 16  # From network architecture (4 pooling layers)
    window_size = (40, 100)  # Original classifier input size
    
    for filename, image in dictionary_of_images.items():
        # Convert to tensor with batch and channel dims
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            output = detection_model(image_tensor)
        
        # Get probabilities for car class
        car_probs = torch.softmax(output, dim=1)[0, 1]  # Shape: [H_heat, W_heat]
        
        # Generate bounding boxes
        boxes = []
        for i in range(car_probs.shape[0]):
            for j in range(car_probs.shape[1]):
                confidence = car_probs[i, j].item()
                
                # Calculate coordinates using fixed network stride
                y = i * effective_stride
                x = j * effective_stride
                
                boxes.append([y, x, window_size[0], window_size[1], confidence])
        
        detections[filename] = np.array(boxes) if boxes else np.zeros((0, 5))

    return detections


# =============================== 5 IoU ========================================
def calc_iou(first_bbox, second_bbox):
    """
    :param first bbox: bbox in format (row, col, n_rows, n_cols)
    :param second_bbox: bbox in format (row, col, n_rows, n_cols)
    :return: iou measure for two given bboxes
    """
    # your code here \/
    first=np.array(first_bbox)
    first[2:]+=first[:2]
    second=np.array(second_bbox)
    second[2:]+=second[:2]
    a=np.min([first[2:],second[2:]],axis=0)
    b=np.max([first[:2],second[:2]],axis=0)
    mutual=np.prod(np.max([a-b,np.zeros(2)],axis=0))
    merge=np.prod(first[2:]-first[:2])+np.prod(second[2:]-second[:2])-mutual
    # print(mutual,merge)
    return mutual/merge
    # your code here /\


# =============================== 6 AUC ========================================
def calc_auc(pred_bboxes, gt_bboxes):
    """
    :param pred_bboxes: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param gt_bboxes: dict of bboxes in format {filenames: bboxes}. bboxes is a
        list of tuples in format (row, col, n_rows, n_cols)
    :return: auc measure for given detections and gt
    """

    all_detections = []
    total_gt = 0

    for filename in gt_bboxes:
        total_gt += len(gt_bboxes[filename])

    for filename in pred_bboxes:
        dets = pred_bboxes[filename]
        sorted_dets = sorted(dets, key=lambda x: -x[4])
        gt_boxes = [tuple(bbox) for bbox in gt_bboxes.get(filename, [])]
        used = [False] * len(gt_boxes)

        for det in sorted_dets:
            row, col, h, w, conf = det
            pred_box = (row, col, h, w)
            max_iou = -1
            best_idx = -1
            for idx, gt_box in enumerate(gt_boxes):
                if used[idx]:
                    continue
                iou = calc_iou(pred_box, gt_box)
                if iou >= 0.5 and iou > max_iou:
                    max_iou = iou
                    best_idx = idx
            if best_idx != -1:
                all_detections.append((conf, 1, det, filename))
                used[best_idx] = True
            else:
                all_detections.append((conf, 0, det, filename))

    all_detections.sort(key=lambda x: (-x[0], -x[1]))

    grouped_detections = []
    current_conf = None
    current_group = []
    for det in all_detections:
        conf, is_tp, index, file = det
        if conf != current_conf:
            if current_conf is not None:
                grouped_detections.append((current_conf, current_group))
            current_conf = conf
            current_group = []
        current_group.append(det)
    if current_conf is not None:
        grouped_detections.append((current_conf, current_group))

    tp_cum = 0
    fp_cum = 0
    precisions = [1.0]  
    recalls = [0.0]

    for conf, group in grouped_detections:
        group_tp = sum(1 for d in group if d[1] == 1)
        group_fp = len(group) - group_tp

        tp_cum += group_tp
        fp_cum += group_fp

        precision = tp_cum / (tp_cum + fp_cum) if (tp_cum + fp_cum) > 0 else 0
        recall = tp_cum / total_gt if total_gt > 0 else 0

        precisions.append(precision)
        recalls.append(recall)

    auc = 0.0
    for i in range(1, len(recalls)):
        recall_prev = recalls[i-1]
        recall_curr = recalls[i]
        precision_prev = precisions[i-1]
        precision_curr = precisions[i]
        auc += (recall_curr - recall_prev) * (precision_prev + precision_curr) / 2

    return auc



# =============================== 7 NMS ========================================
def nms(detections_dictionary, iou_thr=0.37):
    """
    :param detections_dictionary: dict of bboxes in format {filename: detections}
        detections is a N x 5 array, where N is number of detections. Each
        detection is described using 5 numbers: [row, col, n_rows, n_cols,
        confidence].
    :param iou_thr: IoU threshold for nearby detections
    :return: dict in same format as detections_dictionary where close detections
        are deleted
    """

    filenames = detections_dictionary.keys()
    for filename in filenames:
        detections = detections_dictionary[filename]
        sorted_detections = sorted(detections, key=lambda x: -x[4])
        clean_detections = []
        mask = [False] * len(sorted_detections)
        for i in range(len(sorted_detections)):
            if mask[i]:
                continue
            clean_detections.append(sorted_detections[i])
            for j in range(i,len(sorted_detections)):
                if mask[j]:
                    continue
                if calc_iou(sorted_detections[i][:4],sorted_detections[j][:4])>iou_thr:
                    mask[j] = True
        detections_dictionary[filename] = clean_detections
    return detections_dictionary

