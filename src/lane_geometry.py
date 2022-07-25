from copy import copy
import cv2
import numpy as np
from more_itertools import collapse
from shapely import geometry
from more_itertools import grouper
import json

def draw_line(*, img, x_series, y_series, color, width=5):
    # -> Recast the x and y points into usable format for cv2.fillPoly()
    pts = np.vstack((x_series, y_series)).astype(np.int32).T
    # -> Draw the lane onto the image
    cv2.polylines(img, [pts], False, color=color, thickness=width)
    return img

class PointSelf(dict):
    """ Point class represents and manipulates x,y coords. """
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getitem__(self, item):
        return self.get(item, None)

    def __getattr__(self, item):
        return self.get(item, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_()

    def normalize_(self):
        """ Create a new point at the origin """
        for key in ['x', 'y', 'prob','score']:
            if key in self:
                self[key] = float(self[key])

    def __sub__(self, other):
        return PointSelf(x=self.x - other.x, y=self.y - other.y)

    def __abs__(self):
        return np.sqrt(self.x ** 2 + self.y ** 2)

    def __lt__(self, other):
        return self.score - other.score < 0

    def __repr__(self):
        return f'({self.x:.2f}, {self.y:.2f})\n'

    def __str__(self):
        return f'({self.x:.2f}, {self.y:.2f})\n'

    def __copy__(self):
        return self.__class__.from_dict(self)

    @classmethod
    def from_dict(cls, pt):
        return cls(**pt)

    @classmethod
    def from_tuple(cls, pt):
        return cls(x=pt[0], y=pt[1])

    def to_tuple(self):
        return self.x, self.y

    def to_dict(self):
        return self

    def to_str(self):
        return f'({self.x:.4f}, {self.y:.4f})-(score:{self.score:.4f})'

class FloatLengthLine(object):
    def __init__(self, width=None, height=None):
        self.width = width
        self.height = height
        self.represent_points = []
    def __len__(self):
        return len(self.represent_points)

    def append(self, pt):
        self.represent_points.append(pt)

    def clear(self):
        self.represent_points.clear()

    def reverse(self):
        self.represent_points.reverse()

    def thresh(self, thresh):
        ret_elements = copy(self)
        ret_elements.clear()
        for item_spec in self.represent_points:
            if item_spec.score >= thresh:
                ret_elements.append(item_spec)
        return ret_elements

    def center_point(self):
        x_series = [point.x for point in self.represent_points]
        y_series = [point.y for point in self.represent_points]
        score_series = [point.score for point in self.represent_points]
        ret = PointSelf(x=sum(x_series) / len(x_series),
                        y=sum(y_series) / len(y_series),
                        score=sum(score_series) / len(score_series))
        return ret

    def padding_(self, top=0, bottom=0, left=0, right=0):
        self.width = self.width + left + right
        self.height = self.height + top + bottom
        for pt in self.represent_points:
            pt.x = pt.x + left
            pt.y = pt.y + top

    def expand_(self, width, height):
        x_factor = width / self.width
        y_factor = height / self.height
        self.height = height
        self.width = width
        for pt in self.represent_points:
            pt.x = pt.x * x_factor
            pt.y = pt.y * y_factor

    def expand(self, width, height):
        ret = self.__class__(width=width, height=height)
        x_factor = width / self.width
        y_factor = height / self.height
        for pt in self.represent_points:
            pt_exp = copy(pt)
            pt_exp.x = pt.x * x_factor
            pt_exp.y = pt.y * y_factor
            ret.append(pt_exp)
        return ret

    @property
    def score(self):
        return float(np.array([pt.score for pt in self.represent_points]).mean())

    def gen_aux_(self):
        self.data = np.zeros(self.height, dtype=np.float)
        self.mask = np.zeros(self.height, dtype=np.bool)
        ret_line_string = self.to_LineString()
        for y in list(range(0, self.height, 10)):
            current_line = geometry.LineString([(-10, y), (self.width + 10, y)])
            pts = ret_line_string.intersection(current_line)
            if not pts.is_empty:
                if isinstance(pts, geometry.Point):
                    self.data[y] = np.array(pts)[0]
                    self.mask[y] = True
                elif isinstance(pts, geometry.MultiPoint):
                    pts = collapse([pts], geometry.Point)
                    self.data[y] = np.array([np.array(pt)[0] for pt in pts]).mean()
                    self.mask[y] = True
        return self

    def __add__(self, other):
        ret = copy(self)
        for item in other.represent_points:
            ret.append(item)
        return ret

    def __sub__(self, other):
        ret = self.__class__(width=self.width, height=self.height)
        ret.data = (self.data - other.data)
        ret.mask = self.mask * other.mask
        return ret

    def __abs__(self):
        ret = self.data[self.mask]
        if len(ret) > 0:
            return abs(ret).mean()
        else:
            return np.finfo(np.float32).max

    def __eq__(self, other):
        error = self - other
        return abs(error) < 2

    def __lt__(self, other):
        return self.score - other.score < 0

    def strip_(self, thresh):
        ret = copy(self)
        self.clear()
        indexes = np.where([pt.score > thresh for pt in ret])[0]
        if len(indexes) < 2:
            return
        for index_spec in range(min(indexes), max(indexes) + 1):
            self.append(ret[index_spec])

    def fit(self):
        raise NotImplementedError

    def to_Tuple_List(self):
        return [(pt.x, pt.y) for pt in self.represent_points]

    def to_LineString(self):
        return geometry.LineString([(pt.x, pt.y) for pt in self.represent_points])

    def from_CULaneStruct(self, CULaneStruct):
        pass

    def from_CurveLanesStruct(self, CurveLanesStruct):
        pass

    def render_on_image(self, src_image, color=None):
        if color is None:
            color_curr_right = (0, 0, 255)
            color_curr_left = (255, 0, 0)
            color_all = (0, 255, 0)
            color = color_curr_left
        thickness = np.clip(self.height * 0.005, a_min=1, a_max=None).astype(np.int32)
        str_score = f'prob:{self.score:.3}'
        # for p in self:
        #     cv2.circle(src_image, (int(p.x), int(p.y)), 2, color=(255, 0, 0), thickness=thickness)
        x_series = [pt.x for pt in self.represent_points]
        y_series = [pt.y for pt in self.represent_points]
        img = draw_line(img=src_image, x_series=x_series, y_series=y_series,
                        color=color, width=thickness)
        y = int(sum(y_series) / len(y_series))
        x = int(sum(x_series) / len(x_series))
        cv2.putText(src_image, str_score, (x, y), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.0, (200, 200, 250))
        return img

    def _iou(self, lane2):
        new_height = self.height
        new_width = self.width
        lane_width = 15
        im1 = np.zeros((new_height, new_width), np.uint8)
        im2 = np.zeros((new_height, new_width), np.uint8)
        interp_lane1 = self.represent_points
        interp_lane2 = lane2.represent_points
        for i in range(0, len(interp_lane1) - 1):
            cv2.line(im1, (int(interp_lane1[i]['x']), int(interp_lane1[i]['y'])),
                     (int(interp_lane1[i + 1]['x']), int(interp_lane1[i + 1]['y'])), 255, lane_width)
        for i in range(0, len(interp_lane2) - 1):
            cv2.line(im2, (int(interp_lane2[i]['x']), int(interp_lane2[i]['y'])),
                     (int(interp_lane2[i + 1]['x']), int(interp_lane2[i + 1]['y'])), 255, lane_width)
        union_im = cv2.bitwise_or(im1, im2)
        union_sum = union_im.sum()
        intersection_sum = im1.sum() + im2.sum() - union_sum
        if union_sum == 0:
            return 0
        else:
            return intersection_sum / float(union_sum)


def load_lines(file_path):
    with open(file_path) as f:
        target_lines = list(map(str.strip, f))
    return target_lines

def load_json(file_path):
    with open(file_path) as f:
        target_dict = json.load(f)
    return target_dict

def load_CULaneFile(filename, image_shape):
    lanes = load_lines(filename)

    curvelanes = []
    for lane in lanes:
        curvelane = [{'x': x, 'y': y} for x, y in grouper(map(float, lane.split(' ')), 2)]
        curvelanes.append(curvelane)

    height, width = image_shape[0], image_shape[1]
    load_lanes = []
    all_lane_points = []
    per_num_point = []
    for dict_list_line_spec in curvelanes:
        line_spec = FloatLengthLine(width=width, height=height)
        for pt in dict_list_line_spec:
            line_spec.append(PointSelf.from_dict({**pt, 'score': 1.0}))

            all_lane_points.append(pt['x'])
            all_lane_points.append(pt['y'])

        per_num_point.append(len(dict_list_line_spec))
        load_lanes.append(line_spec)
    return load_lanes, all_lane_points, per_num_point

def to_LineStringsStruct(lanes):
    return [lane.to_LineString() for lane in lanes]