import h5py
import tqdm
import logging
import numpy as np
import os

from torch.utils.data import DataLoader

confs = {
    'superpoint': {
        'output': 'feats-spp',
        'model': {
            'name': 'spp_det',
        },
        'preprocessing': {
            'grayscale': True,
            'resize_h': 512,
            'resize_w': 512
        },
        'conf': {
            'descriptor_dim': 256,
            'nms_radius': 3,
            'max_keypoints': 4096,
            'keypoint_threshold': 0.6,  # note: renamed from keypoints_threshold to keypoint_threshold
            'remove_borders': 4
        }
    }
}


def spp(img_lists, feature_out, cfg):
    """extract keypoints info by superpoint using ONNX model"""
    from onnx_demo.onnx_models import SuperPointOnnx
    from onnx_demo.datasets.normalized_dataset import NormalizedDataset

    conf = confs[cfg.network.detection]
    # Transform config to match SuperPointOnnx expected keys
    onnx_config = {
        'nms_radius': conf['conf']['nms_radius'],
        'keypoint_threshold': conf['conf']['keypoint_threshold'],
        'max_keypoints': conf['conf']['max_keypoints'],
        'remove_borders': conf['conf'].get('remove_borders', 4)
    }

    # Path to ONNX model - use relative path from this file's location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    onnx_path = os.path.join(script_dir, '..', 'models', 'superpoint.onnx')
    onnx_path = os.path.normpath(onnx_path)
    
    if not os.path.exists(onnx_path):
        # Fallback: try absolute path relative to project root
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
        onnx_path2 = os.path.join(project_root, 'onnx_demo', 'models', 'superpoint.onnx')
        if os.path.exists(onnx_path2):
            onnx_path = onnx_path2
        else:
            raise FileNotFoundError(f"SuperPoint ONNX model not found at {onnx_path} or {onnx_path2}")
    
    model = SuperPointOnnx(onnx_path, config=onnx_config)

    dataset = NormalizedDataset(img_lists, conf['preprocessing'])
    loader = DataLoader(dataset, num_workers=1)

    feature_file = h5py.File(feature_out, 'w')
    logging.info(f'Exporting features to {feature_out}')
    for data in tqdm.tqdm(loader):
        # Convert tensor to numpy array [1, 1, H, W] float32
        inp = data['image'].numpy().astype(np.float32)  # shape: [1, H, W] or [3, H, W]
        # Add batch dimension if missing
        if inp.ndim == 3:
            inp = inp[np.newaxis]  # [1, 1, H, W] for grayscale
        # Run ONNX inference
        pred = model(inp)

        # Add image size to prediction
        pred['image_size'] = data['size'][0].numpy()

        grp = feature_file.create_group(data['path'][0])
        for k, v in pred.items():
            grp.create_dataset(k, data=v)

        del pred

    feature_file.close()
    logging.info('Finishing exporting features.')


def main(img_lists, feature_out, cfg):
    if cfg.network.detection == 'superpoint':
        spp(img_lists, feature_out, cfg)
    else:
        raise NotImplementedError