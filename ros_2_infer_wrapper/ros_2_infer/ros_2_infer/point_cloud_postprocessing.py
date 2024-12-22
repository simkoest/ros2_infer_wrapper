import torch
import torch.nn.functional as F
import numpy as np
import cv2

def point_cloud_decode_SFA(data, cen_offset, direction, z_coor, dim, K=40):
    hm_cen = torch.from_numpy(data)
    cen_offset = torch.from_numpy(cen_offset)
    direction = torch.from_numpy(direction)
    z_coor = torch.from_numpy(z_coor)
    dim = torch.from_numpy(dim)

    hm_cen = _sigmoid(hm_cen)
    cen_offset = _sigmoid(cen_offset)
     
    batch_size, num_classes, height, width = hm_cen.size()

    hm_cen = point_cloud_non_maximum_suppresion(hm_cen)
    scores, inds, clses, ys, xs = _topk(hm_cen, K=K)
    if cen_offset is not None:
        cen_offset = _transpose_and_gather_feat(cen_offset, inds)
        cen_offset = cen_offset.view(batch_size, K, 2)
        xs = xs.view(batch_size, K, 1) + cen_offset[:, :, 0:1]
        ys = ys.view(batch_size, K, 1) + cen_offset[:, :, 1:2]
    else:
        xs = xs.view(batch_size, K, 1) + 0.5
        ys = ys.view(batch_size, K, 1) + 0.5

    direction = _transpose_and_gather_feat(direction, inds)
    direction = direction.view(batch_size, K, 2)
    z_coor = _transpose_and_gather_feat(z_coor, inds)
    z_coor = z_coor.view(batch_size, K, 1)
    dim = _transpose_and_gather_feat(dim, inds)
    dim = dim.view(batch_size, K, 3)
    clses = clses.view(batch_size, K, 1).float()
    scores = scores.view(batch_size, K, 1)

    # (scores x 1, ys x 1, xs x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, ys-1:2, xs-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
    # detections: [batch_size, K, 10]
    detections = torch.cat([scores, xs, ys, z_coor, dim, direction, clses], dim=2)

    return detections

def _topk(data, K=40):
    scores = data
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (torch.floor_divide(topk_inds, width)).float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (torch.floor_divide(topk_ind, K)).int()
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

def _gather_feat(data, ind, mask=None):
    feat = data
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _transpose_and_gather_feat(data, ind):
    feat = data
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat

def point_cloud_non_maximum_suppresion(data, kernel=3):
    heat = data
    pad =(kernel -1) //2
    hmax = F.max_pool2d(heat, (kernel,kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()    

    return heat*keep

def _sigmoid(x):
    return torch.clamp(x.sigmoid_(), min=1e-4, max=1 - 1e-4)

def point_cloud_post_process_SFA(data, bound_size_x, bound_size_y, bev_height, bev_width, num_classes=3, down_ratio=4, peak_thresh=0.2):
    """
    :param detections: [batch_size, K, 10]
    # (scores x 1, xs x 1, ys x 1, z_coor x 1, dim x 3, direction x 2, clses x 1)
    # (scores-0:1, xs-1:2, ys-2:3, z_coor-3:4, dim-4:7, direction-7:9, clses-9:10)
    :return:
    """
    # TODO: Need to consider rescale to the original scale: x, y
    detections = data.cpu().numpy().astype(np.float32)            
    ret = []
    for i in range(detections.shape[0]):
        top_preds = {}
        classes = detections[i, :, -1]
        for j in range(num_classes):
            inds = (classes == j)
            # x, y, z, h, w, l, yaw
            top_preds[j] = np.concatenate([
                detections[i, inds, 0:1],
                detections[i, inds, 1:2] * down_ratio,
                detections[i, inds, 2:3] * down_ratio,
                detections[i, inds, 3:4],
                detections[i, inds, 4:5],
                detections[i, inds, 5:6] / bound_size_y * bev_width,
                detections[i, inds, 6:7] / bound_size_x * bev_height,
                _get_yaw(detections[i, inds, 7:9]).astype(np.float32)], axis=1)
            # Filter by peak_thresh
            if len(top_preds[j]) > 0:
                keep_inds = (top_preds[j][:, 0] > peak_thresh)
                top_preds[j] = top_preds[j][keep_inds]
        ret.append(top_preds)

    return ret[0]

def _get_yaw(direction):
    return np.arctan2(direction[:, 0:1], direction[:, 1:2])

# def draw_predictions_SFA(data, img, colors, num_classes=3):
#     detections = data
#     for j in range(num_classes):
#         if len(detections[j]) > 0:
#             for det in detections[j]:
#                 # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
#                 _score, _x, _y, _z, _h, _w, _l, _yaw = det
#                 drawRotatedBox(img, _x, _y, _w, _l, _yaw, colors[int(j)])

#     return img

# def _draw_rotated_box(img, x, y, w, l, yaw, color):
#     bev_corners = get_corners(x, y, w, l, yaw)
#     corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
#     cv2.polylines(img, [corners_int], True, color, 2)
#     corners_int = bev_corners.reshape(-1, 2).astype(int)
#     cv2.line(img, (corners_int[0, 0], corners_int[0, 1]), (corners_int[3, 0], corners_int[3, 1]), (255, 255, 0), 2)

def _get_corners(x, y, w, l, yaw):
    bev_corners = np.zeros((4, 2), dtype=np.float32)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    # front left
    bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw

    # rear left
    bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw

    # rear right
    bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw
    bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw

    # front right
    bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw
    bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw

    return bev_corners

def point_cloud_draw_predictions_SFA(data, RGB_Map, colors, bev_width, bev_height, num_classes=3):
    model_inputs = torch.from_numpy(np.array(RGB_Map[0]))        
    bev_map = (model_inputs.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    
    detections = data
    
    bev_map = cv2.resize(bev_map, (bev_width, bev_height))    

    for j in range(num_classes):
        if len(detections[j]) > 0:
            for det in detections[j]:
                # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
                _score, _x, _y, _z, _h, _w, _l, _yaw = det
                _draw_rotated_box(bev_map, _x, _y, _w, _l, _yaw, colors[int(j)])
    
    return bev_map

def _draw_rotated_box(img, x, y, w, l, yaw, color):
    bev_corners = _get_corners(x, y, w, l, yaw)
    corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
    cv2.polylines(img, [corners_int], True, color, 2)
    corners_int = bev_corners.reshape(-1, 2).astype(int)
    cv2.line(img, (corners_int[0, 0], corners_int[0, 1]), (corners_int[3, 0], corners_int[3, 1]), (255, 255, 0), 2)