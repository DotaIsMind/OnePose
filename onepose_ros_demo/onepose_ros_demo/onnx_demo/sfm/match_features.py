import h5py
import logging
import tqdm
import os.path as osp
import numpy as np

confs = {
    'superglue': {
        'output': 'matches-spg',
        'conf': {
            'descriptor_dim': 256,
            'weights': 'outdoor',
            'match_threshold': 0.7
        }
    }
}


def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))


def _resolve_superglue_onnx_path(cfg):
    """Resolve superglue.onnx path (same layout as extract_features SuperPoint)."""
    net = cfg.network
    p = getattr(net, 'matching_model_path', None)
    if p and str(p).lower().endswith('.onnx') and osp.exists(p):
        return osp.normpath(p)

    script_dir = osp.dirname(osp.abspath(__file__))
    onnx_path = osp.normpath(osp.join(script_dir, '..', 'models', 'superglue.onnx'))
    if osp.exists(onnx_path):
        return onnx_path

    project_root = osp.abspath(osp.join(script_dir, '..', '..', '..'))
    onnx_path2 = osp.join(project_root, 'onnx_demo', 'models', 'superglue.onnx')
    if osp.exists(onnx_path2):
        return onnx_path2

    raise FileNotFoundError(
        f"SuperGlue ONNX not found. Tried: {onnx_path}, {onnx_path2}. "
        "Set cfg.network.matching_model_path to a .onnx file or place superglue.onnx under onnx_demo/models/."
    )


def _features_pair_to_sg_data(feats0, feats1):
    """Build the dict expected by SuperGlueOnnx from two HDF5 feature groups."""
    kpts0 = np.asarray(feats0['keypoints'], dtype=np.float32)
    kpts1 = np.asarray(feats1['keypoints'], dtype=np.float32)
    desc0 = np.asarray(feats0['descriptors'], dtype=np.float32)
    desc1 = np.asarray(feats1['descriptors'], dtype=np.float32)
    sc0 = np.asarray(feats0['scores'], dtype=np.float32)
    sc1 = np.asarray(feats1['scores'], dtype=np.float32)

    sz0 = np.asarray(feats0['image_size']).reshape(-1)
    sz1 = np.asarray(feats1['image_size']).reshape(-1)
    # Legacy PyTorch path used (1,1) + image_size[::-1] for placeholder images
    sh0 = (1, 1) + tuple(sz0)[::-1]
    sh1 = (1, 1) + tuple(sz1)[::-1]

    return {
        'keypoints0':   kpts0[np.newaxis, ...],
        'keypoints1':   kpts1[np.newaxis, ...],
        'descriptors0': desc0[np.newaxis, ...],
        'descriptors1': desc1[np.newaxis, ...],
        'scores0':      sc0[np.newaxis, ...],
        'scores1':      sc1[np.newaxis, ...],
        'image0':       np.empty(sh0, dtype=np.float32),
        'image1':       np.empty(sh1, dtype=np.float32),
    }


def _pred_tensor_to_numpy(t):
    if hasattr(t, 'detach'):
        return t.detach().cpu().numpy()
    if hasattr(t, 'numpy'):
        return t.numpy()
    return np.asarray(t)


def spg(cfg, feature_path, covis_pairs, matches_out, vis_match=False):
    """Match features with SuperGlue via ONNX Runtime (SuperGlueOnnx)."""
    from onnx_demo.onnx_models import SuperGlueOnnx
    from onnx_demo.utils.vis_utils import vis_match_pairs

    onnx_path = _resolve_superglue_onnx_path(cfg)
    model = SuperGlueOnnx(onnx_path)

    assert osp.exists(feature_path), feature_path
    feature_file = h5py.File(feature_path, 'r')
    logging.info(f'Exporting matches to {matches_out}')

    with open(covis_pairs, 'r') as f:
        pair_list = f.read().rstrip('\n').split('\n')

    match_file = h5py.File(matches_out, 'w')
    matched = set()
    for pair in tqdm.tqdm(pair_list):
        name0, name1 = pair.split(' ')
        pair = names_to_pair(name0, name1)

        if len({(name0, name1), (name1, name0)} & matched) \
                or pair in match_file:
            continue

        feats0, feats1 = feature_file[name0], feature_file[name1]
        data = _features_pair_to_sg_data(feats0, feats1)
        pred = model(data)
        n0 = int(feats0['keypoints'].shape[0])
        n1 = int(feats1['keypoints'].shape[0])

        grp = match_file.create_group(pair)
        m0 = _pred_tensor_to_numpy(pred['matches0'][0])
        grp.create_dataset('matches0', data=m0.astype(np.int16))

        m1 = _pred_tensor_to_numpy(pred['matches1'][0])
        grp.create_dataset('matches1', data=m1.astype(np.int16))

        scores0 = _pred_tensor_to_numpy(pred['matching_scores0'][0])
        grp.create_dataset('matching_scores0', data=scores0.astype(np.float16))
        scores1 = _pred_tensor_to_numpy(pred['matching_scores1'][0])
        grp.create_dataset('matching_scores1', data=scores1.astype(np.float16))

        matched |= {(name0, name1), (name1, name0)}

        if vis_match and n0 > 0 and n1 > 0:
            vis_match_pairs(pred, feats0, feats1, name0, name1)

    match_file.close()
    feature_file.close()
    logging.info('Finishing exporting matches.')


def main(cfg, feature_out, covis_pairs_out, matches_out, vis_match=False):
    if cfg.network.matching == 'superglue':
        spg(cfg, feature_out, covis_pairs_out, matches_out, vis_match)
    else:
        raise NotImplementedError
