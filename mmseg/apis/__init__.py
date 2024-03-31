# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

from .inference import inference_segmentor, init_segmentor, show_result_pyplot
from .test import multi_gpu_test, single_gpu_test
    # t_sne_test, draw_uncertainty, calculate_logme, get_model_complexity, \
    # calculate_norm, draw_unlabeled
from .train import get_root_logger, set_random_seed, train_segmentor

__all__ = [
    'get_root_logger', 'set_random_seed', 'train_segmentor', 'init_segmentor',
    'inference_segmentor', 'multi_gpu_test', 'single_gpu_test',
    'show_result_pyplot'
]
