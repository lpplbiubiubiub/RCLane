import time
import os

from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper
from src.model_utils.device_adapter import get_device_id, get_device_num
from src.culane_dataset import data_to_mindrecord_byte_image, create_culane_dataset
from mindspore import context, Tensor, Parameter
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.context import ParallelMode
from mindspore.common import set_seed

set_seed(1)

def create_mindrecord_dir(prefix, mindrecord_dir):
    if not os.path.isdir(mindrecord_dir):
        os.makedirs(mindrecord_dir)
    if config.dataset == "culane":
        if os.path.isdir(config.culane_root):
            print("Create Mindrecord.")
            data_to_mindrecord_byte_image("culane", True, prefix)
            print("Create Mindrecord Done, at {}".format(mindrecord_dir))
        else:
            raise Exception("culane_root not exits.")
    else:
        if os.path.isdir(config.IMAGE_DIR) and os.path.exists(config.ANNO_PATH):
            print("Create Mindrecord.")
            data_to_mindrecord_byte_image("other", True, prefix)
            print("Create Mindrecord Done, at {}".format(mindrecord_dir))
        else:
            raise Exception("IMAGE_DIR or ANNO_PATH not exits.")
    # while not os.path.exists(mindrecord_file+".db"):
    #     time.sleep(5)

# @moxing_wrapper(pre_process=modelarts_pre_process)
def train_maskrcnn():
    device_target = config.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target, device_id=get_device_id())

    config.mindrecord_dir = os.path.join(config.culane_root, config.mindrecord_dir)
    print('\ntrain.py config:\n', config)
    print("Start train for relaychain!")

    dataset_sink_mode_flag = True
    if not config.do_eval and config.run_distribute:
        init()
        rank = get_rank()
        dataset_sink_mode_flag = device_target == 'Ascend'
        device_num = get_group_size()
        context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                          gradients_mean=True)
    else:
        rank = 0
        device_num = 1

    print("Start create dataset!")

    # It will generate mindrecord file in config.mindrecord_dir,
    # and the file name is MaskRcnn.mindrecord0, 1, ... file_num.
    prefix = "culane.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    if rank == 0 and not os.path.exists(mindrecord_file):
        create_mindrecord_dir(prefix, mindrecord_dir)
    # create_mindrecord_dir(prefix, mindrecord_dir)

    if not config.only_create_dataset:
        # loss_scale = float(config.loss_scale)
        # When create MindDataset, using the fitst mindrecord file, such as MaskRcnn.mindrecord0.
        dataset = create_culane_dataset(mindrecord_file, batch_size=config.batch_size,
                                        device_num=device_num, rank_id=rank)
        print('dataset_finish')
        dataset_size = dataset.get_dataset_size()
        print("total images num: ", dataset_size)
        print("Create dataset done!")




if __name__ == '__main__':
    train_maskrcnn()