# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.17.0

from .builder import DATASETS
from .cityscapes import CityscapesDataset


@DATASETS.register_module()
class Bdd100kDataset(CityscapesDataset):
    """DarkZurichDataset dataset."""

    def __init__(self, **kwargs):
        super(Bdd100kDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            **kwargs)
