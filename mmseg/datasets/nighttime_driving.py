# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.17.0

from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class NighttimeDataset(CityscapesDataset):
    """DarkZurichDataset dataset."""

    def __init__(self, **kwargs):
        super(NighttimeDataset, self).__init__(
            img_suffix='_leftImg8bit.png',
            seg_map_suffix='_gtCoarse_labelTrainIds.png',
            **kwargs)
