# Copyright 2021-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""create train or eval dataset."""

import os
import cv2
import warnings
import numpy as np
import mindspore.dataset as de
import mindspore.dataset.vision.c_transforms as C
from .model_utils.config import config
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.lines import LineString as ia_LineString
from imgaug.augmentables.lines import LineStringsOnImage
from .lane_augmentation import get_fastdraw_aug, get_infer_aug
from .lane_geometry import FloatLengthLine, PointSelf, load_CULaneFile, load_lines, to_LineStringsStruct
from mindspore.mindrecord import FileWriter
from .lane_codec import encode


if config.device_target == "Ascend":
    np_cast_type = np.float16
else:
    np_cast_type = np.float32

def resize_by_wh(*, img, width, height):
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_LINEAR)
    return resized

def create_culane_label(is_training):
    culane_root = config.culane_root
    data_type = config.val_data_type
    if is_training:
        data_type = config.train_data_type

    images_list_path = os.path.join(culane_root, 'list', '{}.txt'.format(data_type))
    images_list = load_lines(images_list_path)
    images_annos_path = []
    images_num = len(images_list)
    for ind, img_list in enumerate(images_list):
        path_pair = dict(
            image_path=os.path.join(culane_root, img_list[1:]),
            anno_path=os.path.join(culane_root, img_list[1:].replace('.jpg', '.lines.txt')))
        images_annos_path.append(path_pair)

        if (ind + 1) % 10 == 0:
            print("{}/{}: parsing annotation for image={}".format(ind + 1, images_num, images_list))
    return images_annos_path

def data_to_mindrecord_byte_image(dataset="culane", is_training=True, prefix="culane.mindrecord", file_num=1):
    """Create MindRecord file."""
    mindrecord_dir = config.mindrecord_dir
    mindrecord_path = os.path.join(mindrecord_dir, prefix)

    writer = FileWriter(mindrecord_path, file_num, overwrite=True)
    if dataset == "culane":
        images_annos = create_culane_label(is_training)
    else:
        print("Error unsupported other dataset")
        return

    relaychain_json = {
        "image": {"type": "bytes"},
        "all_points": {"type": "int32", 'shape': [-1]},
        "points_onelane": {"type": "int32", 'shape': [-1]}
    }

    writer.add_schema(relaychain_json, "culane_json")

    image_files_num = len(images_annos)
    for ind, image_anno in enumerate(images_annos):
        image_path, anno_path = image_anno['image_path'], image_anno['anno_path']
        with open(image_path, 'rb') as f:
            img = f.read()

        image_shape = (config.image_height, config.image_width)
        anno_lanes, all_points, per_num = load_CULaneFile(anno_path, image_shape)
        all_points = np.array(all_points, dtype=np.int32)
        per_num = np.array(per_num, dtype=np.int32)
        row = {"image": img, "all_points": all_points, 'points_onelane': per_num}
        if (ind + 1) % 10 == 0:
            print("writing {}/{} into mindrecord".format(ind + 1, image_files_num))
        writer.write_raw_data([row])

    writer.commit()


def preprocess_fn(image, all_points, points_onelane, is_training):
    """Data augmentation function."""

    resize_height = config.resize_height
    resize_width = config.resize_width

    image_bgr = image.copy()
    image_bgr[:, :, 0] = image[:, :, 2]
    image_bgr[:, :, 1] = image[:, :, 1]
    image_bgr[:, :, 2] = image[:, :, 0]
    image_shape = image_bgr.shape[:2]

    lanes_tuple = []
    start = 0
    for num in points_onelane:
        num_point = int(num*2)
        lane = []
        lane_points = all_points[start:num_point+start]
        for i in range(0, len(lane_points), 2):
            lane.append((float(lane_points[i]), float(lane_points[i+1])))
        start += num_point
        lanes_tuple.append(lane)

    lss = [ia_LineString(lane_tuple) for lane_tuple in lanes_tuple]

    if is_training:
        aug = get_fastdraw_aug()
        lsoi = LineStringsOnImage(lss, shape=image_shape)
        batch = ia.Batch(images=[image_bgr], line_strings=[lsoi])
        batch_aug = list(aug.augment_batches([batch]))[0]  # augment_batches returns a generator
        image_aug = batch_aug.images_aug[0]
        lsoi_aug = batch_aug.line_strings_aug[0]

        new_image = cv2.resize(image_aug, (800, 320))
        new_image = new_image.astype(np_cast_type)
        new_lanes = []
        for shapely_line in lsoi_aug:
            line_spec = FloatLengthLine(width=image_shape[1], height=image_shape[0])
            for kpt in shapely_line.to_keypoints():
                line_spec.append(PointSelf(x=kpt.x, y=kpt.y, score=1.0))
            line_spec.expand_(resize_width, resize_height)
            new_lanes.append(line_spec)
        new_lanes = to_LineStringsStruct(new_lanes)

        segment, up_arrow, down_arrow, up_bound, down_bound = encode(new_lanes)
        print(segment.shape, up_arrow.shape)

        return new_image, segment, up_arrow, down_arrow, up_bound, down_bound

    else:
        aug = get_infer_aug()
        lss = []
        lsoi = LineStringsOnImage(lss, shape=image_shape)
        batch = ia.Batch(images=[image], line_strings=[lsoi])
        batch_aug = list(aug.augment_batches([batch]))[0]
        image_aug = batch_aug.images_aug[0]

        new_image = resize_by_wh(img=image_aug, width=resize_width, height=resize_height)
        new_image = new_image.astype(np_cast_type)
        # new_lanes = ['infer_no_aug']

        return new_image


def create_culane_dataset(mindrecord_file,
                          batch_size=2,
                          device_num=1,
                          rank_id=0,
                          is_training=True,
                          num_parallel_workers=8):
    cv2.setNumThreads(0)
    de.config.set_prefetch_size(8)
    ds = de.MindDataset(mindrecord_file, columns_list=["image", "all_points", 'points_onelane'],
                        num_shards=device_num, shard_id=rank_id,
                        num_parallel_workers=1, shuffle=is_training)

    decode = C.Decode()
    ds = ds.map(operations=decode, input_columns=["image"])
    compose_map_func = (lambda image, all_points, points_onelane:
                        preprocess_fn(image, all_points, points_onelane, is_training))

    if is_training:
        ds = ds.map(operations=compose_map_func,
                    input_columns=["image", "all_points", 'points_onelane'],
                    output_columns=["image", "segment", "up_arrow", "down_arrow", "up_bound", "down_bound"],
                    column_order=["image", "segment", "up_arrow", "down_arrow", "up_bound", "down_bound"],
                    python_multiprocessing=False,
                    num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True, pad_info=None)


    else:
        ds = ds.map(operations=compose_map_func,
                    input_columns=["image", "all_points", 'points_onelane'],
                    output_columns=["image"],
                    column_order=["image",],
                    num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)

    return ds
