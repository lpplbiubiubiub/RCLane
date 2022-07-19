import numpy as np
import shapely.speedups
from more_itertools import collapse
from shapely.geometry import Point, CAP_STYLE, JOIN_STYLE, MultiPoint, LineString
from .lane_geometry import FloatLengthLine, PointSelf
import cv2
from copy import copy
from .model_utils.config import config

shapely.speedups.enable()


def draw_line(img, x_series, y_series, color, width=5):
    # -> Recast the x and y points into usable format for cv2.fillPoly()
    pts = np.vstack((x_series, y_series)).astype(np.int32).T
    # -> Draw the lane onto the image
    cv2.polylines(img, [pts], False, color=color, thickness=width)
    return img

def lanes_to_mask(lanes, fm_height, fm_width, line_width=5):
    lane_mask = np.zeros((fm_height, fm_width)).astype('uint8')
    for lane_spec in lanes:
        lane_mask = draw_line(lane_mask,
                              x_series=lane_spec.coords.xy[0],
                              y_series=lane_spec.coords.xy[1],
                              color=255, width=line_width)
    data = (lane_mask > 10).astype(np.float32)
    return data

def get_up_down_arrows(intersect_points, current_point):
    ins_pts = collapse(intersect_points, base_type=Point)
    vectors = [np.array(pt_spec) - np.array(current_point) for pt_spec in ins_pts]
    up_arrow = min(vectors, key=lambda x: x[1])
    down_arrow = max(vectors, key=lambda x: x[1])
    return up_arrow, down_arrow


def encode(lanes):
    fm_height = config.fm_height
    fm_width = config.fm_width
    seg_threshold = config.seg_threshold
    line_width = config.line_width
    step_length = config.step_length

    semantic_fine = np.zeros((fm_height, fm_width))
    up_arrow = np.zeros((fm_height, fm_width, 2))
    down_arrow = np.zeros((fm_height, fm_width, 2))
    up_bound = np.zeros((fm_height, fm_width, 2))
    down_bound = np.zeros((fm_height, fm_width, 2))

    if len(lanes) == 0:
        return
    semantic_coarse = lanes_to_mask(lanes, fm_height, fm_width, line_width)
    foreground = np.where(semantic_coarse > seg_threshold)
    for (y_spec, x_spec) in zip(foreground[0], foreground[1]):
        current_point = Point(x_spec, y_spec)
        current_envelop = current_point.buffer(distance=step_length, resolution=100, quadsegs=None,
                                               cap_style=CAP_STYLE.round, join_style=JOIN_STYLE.round,
                                               mitre_limit=5.0, single_sided=False).exterior
        try:
            nearest_line = min(lanes, key=lambda line_x: line_x.distance(current_point))
        except Exception as e:
            print('e', e)
        intersect_points = current_envelop.intersection(nearest_line)
        terminate_points = nearest_line.boundary
        if isinstance(intersect_points, MultiPoint):
            semantic_fine[y_spec, x_spec] = 1
            up_arrow_delta, down_arrow_delta = get_up_down_arrows(intersect_points, current_point)
            up_arrow[y_spec, x_spec] = up_arrow_delta ##(x, y)
            down_arrow[y_spec, x_spec] = down_arrow_delta

            up_bound_delta, down_bound_delta = get_up_down_arrows(terminate_points, current_point)

            up_cycle = Point(np.array(current_point) + up_bound_delta / 2).buffer(
                np.sqrt(((up_bound_delta / 2) ** 2).sum()), cap_style=CAP_STYLE.round)
            down_cycle = Point(np.array(current_point) + down_bound_delta / 2).buffer(
                np.sqrt(((down_bound_delta / 2) ** 2).sum()), cap_style=CAP_STYLE.round)
            try:
                up_bound_length = up_cycle.intersection(nearest_line).length
            except Exception as e:
                print(e)
                up_bound_length = np.sqrt(((up_bound_delta / 2) ** 2).sum())
            up_bound[y_spec, x_spec] = (up_bound_length / 100 + 1, up_bound_length / 100 + 1)

            try:
                down_bound_length = down_cycle.intersection(nearest_line).length
            except Exception as e:
                print(e)
                down_bound_length = np.sqrt(((down_bound_delta / 2) ** 2).sum())
            down_bound[y_spec, x_spec] = (down_bound_length / 100 + 1, down_bound_length / 100 + 1)

    return softmax_gt(semantic_fine, seg_threshold), up_arrow, down_arrow, up_bound, down_bound


def decode_branch(current_x, current_y, semantic_fine, arrow, bound):
    fm_height = config.fm_height
    fm_width = config.fm_width
    seg_threshold = config.seg_threshold
    step_length = config.step_length

    arrow_dx = arrow[..., 0]
    arrow_dy = arrow[..., 1]

    remain_steps = []
    target_lane = FloatLengthLine(width=fm_width, height=fm_height)
    for index in range(fm_height):
        current_score = semantic_fine[current_y, current_x]
        if current_score > seg_threshold:
            remain_steps.append(bound[current_y, current_x] * 100 / step_length + index)
        arrow_delta = (arrow_dx[current_y, current_x], arrow_dy[current_y, current_x])

        current_x = np.floor(
            current_x + arrow_delta[0] / np.sqrt(arrow_delta[0] ** 2 + arrow_delta[1] ** 2) * step_length).astype(int)
        current_y = np.floor(
            current_y + arrow_delta[1] / np.sqrt(arrow_delta[0] ** 2 + arrow_delta[1] ** 2) * step_length).astype(int)

        if (0 <= current_x < fm_width) and (0 <= current_y < fm_height):
            pass
        else:
            break
        current_pt = PointSelf(x=current_x, y=current_y, score=semantic_fine[current_y, current_x])
        target_lane.append(current_pt)
        if len(remain_steps) != 0:
            ret = np.sqrt(sum([i ** 2 for i in remain_steps]) / len(remain_steps))
        else:
            ret = 1
        if semantic_fine[current_y, current_x] > seg_threshold:
            continue
        if index > ret * 0.75:
            break
    return target_lane

def decode(keypoint, semantic_fine, up_arrow, down_arrow, up_bound, down_bound):
    seg_threshold = config.seg_threshold

    lines = []
    foreground = np.where(keypoint > seg_threshold)
    for (y_spec, x_spec) in zip(foreground[0], foreground[1]):
        current_x = np.floor(x_spec).astype(int)
        current_y = np.floor(y_spec).astype(int)
        up_branch = decode_branch(current_x, current_y, semantic_fine, up_arrow, up_bound[..., 0])
        down_branch = decode_branch(current_x, current_y, semantic_fine, down_arrow, down_bound[..., 0])
        up_branch.reverse()
        total_lane = up_branch + down_branch
        if len(total_lane) > 1:
            lines.append(total_lane)

    thresh_lines = thresh_line(lines, 0.10)
    sparse_lines = iou_nms(thresh_lines, 0.5)
    print(len(sparse_lines))
    return sparse_lines


def thresh_line(lines, thresh=0.10):
    result = []
    for line_spec in lines:
        if line_spec.score >= thresh:
            result.append(line_spec)
    return result

def iou_nms(lines, thresh=0.5):
    if len(lines) == 0:
        return copy(lines)
    sorted_lines = sorted(lines, reverse=True)

    selected = [False] * len(sorted_lines)
    result = []
    for i in range(len(selected)):
        if selected[i]:
            continue
        result.append(sorted_lines[i])
        selected[i] = True
        for j in range(i + 1, len(selected)):
            iou = sorted_lines[i]._iou(sorted_lines[j])
            if iou >= thresh:
                selected[j] = True
    return result


def softmax_gt(data, seg_thresh):
    ret_pos = data > seg_thresh
    ret_neg = ~ret_pos
    return np.dstack([ret_neg, ret_pos])


def add_type(data):
    if len(data.shape) == 2:
        ret = data[..., None].astype(np.float32)
    else:
        ret = data.astype(np.float32)
    return ret



