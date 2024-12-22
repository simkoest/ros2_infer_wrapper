import cv2
import numpy as np
from scipy import special
import colorsys
import random
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 

def image_deresize(data, size):
    data = np.array(data, np.uint8)    
    target_height, target_width = data.shape[:2]
    image_height, image_width = size
    scale = min(target_width/image_width, target_height/image_height)
    scaled_width, scaled_height = int(scale * image_width), int(scale * image_height)
    data = data[0:scaled_height, 0:scaled_width]
    data = cv2.resize(data, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
    return data

def image_draw_bounding_boxes(data, boxes, labels, scores, classes, scale_boxes_by_ratio = 1, score_threshold=0.7):
    # Resize boxes
    image = data    
    ratio = scale_boxes_by_ratio / min(image.shape[0], image.shape[1])
    boxes /= ratio

    # _, ax = plt.subplots(1, figsize=(12,9))
    image = np.array(image)
    # ax.imshow(image)
    
    

    # Showing boxes with score > 0.7
    for box, label, score in zip(boxes, labels, scores):
        if score > score_threshold:
            x1 = int(box[0])
            x2 = int(box[2])
            y1 = int(box[1])
            y2 = int(box[3])
            cv2.rectangle(image, (x1, y1), (x2,y2), color=(255,0,0), thickness=2)
            #rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=1, edgecolor='b', facecolor='none')
            cv2.putText(image, classes[label] + ':' + str(np.round(score, 2)), (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36,255,12), 1)
            # ax.annotate(classes[label] + ':' + str(np.round(score, 2)), (box[0], box[1]), color='w', fontsize=12)
            # ax.add_patch(rect)
    
    return image

def image_colorize_image_segmentation(data, rgb_image, color_palette, opacity=1.0):
    if isinstance(rgb_image, Image):
        cv_bridge = CvBridge()
        rgb_image = cv_bridge.imgmsg_to_cv2(rgb_image, desired_encoding="passthrough")
    color_palette = np.array(color_palette)
    color_seg = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
    h, w = data.shape
    for i in range(h):
        for j in range(w):
            color_seg[i, j] = color_palette[data[i, j]]
    new_image = rgb_image * (1 - opacity) + color_seg * opacity
    new_image = new_image.astype(np.uint8)
    return new_image


def image_define_anchor_boxes(data, anchors, strides, xyscale=[1,1,1]):    
    '''define anchor boxes'''
    pred_bbox = data
    anchors = np.array(anchors)
    strides = np.array(strides)

    for i, pred in enumerate(pred_bbox):
        conv_shape = pred.shape
        output_size = conv_shape[1]
        conv_raw_dxdy = pred[:, :, :, :, 0:2]
        conv_raw_dwdh = pred[:, :, :, :, 2:4]
        xy_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
        xy_grid = np.expand_dims(np.stack(xy_grid, axis=-1), axis=2)

        xy_grid = np.tile(np.expand_dims(xy_grid, axis=0), [1, 1, 1, 3, 1])
        xy_grid = xy_grid.astype(float)

        pred_xy = ((special.expit(conv_raw_dxdy) * xyscale[i]) - 0.5 * (xyscale[i] - 1) + xy_grid) * strides[i]
        pred_wh = (np.exp(conv_raw_dwdh) * anchors[i])
        pred[:, :, :, :, 0:4] = np.concatenate([pred_xy, pred_wh], axis=-1)

    pred_bbox = [np.reshape(x, (-1, np.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = np.concatenate(pred_bbox, axis=0)
    return pred_bbox

def remove_low_prob_boxes(data, org_img_shape, input_size, score_threshold):
    '''remove boundary boxs with a low detection probability'''
    valid_scale=[0, np.inf]
    input_size = input_size[0]
    pred_bbox = np.array(data)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # # (3) clip some boxes that are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

def image_non_maximum_suppression(data, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms

    """
    bboxes = data

    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def image_bboxes_iou(boxes1, boxes2):
    '''calculate the Intersection Over Union value'''
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious

def image_draw_bboxes_to_image(data, orig_image, classes, show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """
    bboxes = data
    num_classes = len(classes)
    image_h, image_w, _ = orig_image.shape
    orig_image = np.ascontiguousarray(orig_image, dtype=np.uint8)
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(orig_image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(orig_image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)
            cv2.putText(orig_image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return orig_image

def image_color_label(data, colors):
    cmap = np.asarray(colors, dtype='uint8')

    return cmap[data]