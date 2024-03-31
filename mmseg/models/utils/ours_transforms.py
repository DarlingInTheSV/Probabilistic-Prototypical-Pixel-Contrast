import torch
import numpy as np
from numpy import random


class RandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """
        dataset = results['dataset']
        assert dataset in ['gta', 'synthia']
        img = results['img']
        proto_sigma = results['proto_sigma']
        if proto_sigma is  None: # test for using CBC
            proto_sigma = torch.tensor([5.6651e-05, 3.7127e-04, 5.8226e-04, 4.8870e-02, 3.2027e-01, 1.5055e-03,
            1.5945e-02, 2.0218e-01, 8.0991e-04, 3.7779e-03, 1.7063e-05, 2.3787e-02,
            9.1081e-03, 5.5943e-04, 2.2627e-02, 1.3021e-01, 1.0346e-01, 4.5112e-02,
            7.0751e-02]).to(img.device)
        else:
            sfm = torch.nn.Softmax()
            proto_sigma = sfm(proto_sigma / 0.01)
        v, i = torch.sort(proto_sigma, descending=True)
        if dataset is 'gta':
            top_k = 10
            top_k_value, top_k_index = v[:top_k], i[:top_k]
        else:
            top_k = 13
            top_k_value, top_k_index = v[:top_k], i[:top_k]
            synthia_ignore = torch.tensor([9, 14, 16]).to(top_k_index.device)  # ignore 3 classes that aren't appeared
            for ign in synthia_ignore:
                top_k_value = top_k_value[ign != top_k_index]
                top_k_index = top_k_index[ign != top_k_index]

        if 'crop_bbox' in results:
            crop_bbox = results['crop_bbox']
        else:
            crop_bbox = self.get_crop_bbox(img)

            best_score = -1
            best_crop_bbox = None
            # Repeat 10 times
            for _ in range(10):
                if best_score >= 0:
                    crop_bbox = self.get_crop_bbox(img)
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = torch.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                score = 0
                # if len(cnt) > 1 and torch.max(cnt).item() / torch.sum(cnt).item() < self.cat_max_ratio:
                #     cnt_valid = cnt[cnt > 1]
                #     score = cnt_valid.float().log().sum().item()
                for l in labels:
                    if l in top_k_index:
                        index_for_top = (l == top_k_index).nonzero(as_tuple=True)[0]
                        index_for_cnt = (l == labels).nonzero(as_tuple=True)[0]
                        score += top_k_value[index_for_top] * cnt[index_for_cnt]
                if score > best_score:
                    best_score = score
                    best_crop_bbox = crop_bbox
            crop_bbox = best_crop_bbox

        # crop the image
        box = ((crop_bbox[2], crop_bbox[0]), crop_bbox[3] - crop_bbox[2], crop_bbox[1] - crop_bbox[0])

        img = self.crop(img, crop_bbox)


        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape
        results['crop_bbox'] = crop_bbox
        results['crop_box'] = box

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


class RandomCropNoProd(RandomCrop):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        super().__init__(crop_size, cat_max_ratio, ignore_index)

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        if 'crop_bbox' in results:
            crop_bbox = results['crop_bbox']
        else:
            crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = torch.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and torch.max(cnt).item() / torch.sum(
                        cnt).item() < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape
        results['crop_bbox'] = crop_bbox

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        return results
