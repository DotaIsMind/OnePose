import os
import h5py
import logging
import tqdm
import subprocess
import os.path as osp
import numpy as np

from pathlib import Path
try:
    from src.utils.colmap.read_write_model import CAMERA_MODEL_NAMES, Image, read_cameras_binary, read_images_binary
    from src.utils.colmap.database import COLMAPDatabase
except ImportError:
    from src.utils.colmap.read_write_model import CAMERA_MODEL_NAMES, Image, read_cameras_binary, read_images_binary
    from src.utils.colmap.database import COLMAPDatabase


def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))


def _colmap_headless_env():
    """Return env vars that force COLMAP CLI into headless mode."""
    env = os.environ.copy()
    env['QT_QPA_PLATFORM'] = 'offscreen'
    env['DISPLAY'] = ''
    env['WAYLAND_DISPLAY'] = ''
    # Avoid inheriting OpenCV Qt plugin path from Python env.
    env.pop('QT_PLUGIN_PATH', None)
    env.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
    return env


def geometric_verification(colmap_path, database_path, pairs_path):
    """ Geometric verfication """
    logging.info('Performing geometric verification of the matches...')
    # Prefer in-process pycolmap when available; fallback to COLMAP CLI.
    try:
        import pycolmap  # type: ignore
    except Exception:
        pycolmap = None

    if pycolmap is not None and hasattr(pycolmap, "verify_matches"):
        try:
            # Newer pycolmap signatures are keyword-based.
            pycolmap.verify_matches(
                database_path=str(database_path),
                pairs_path=str(pairs_path),
            )
            return
        except TypeError:
            # Backward compatibility for positional-only signatures.
            try:
                pycolmap.verify_matches(str(database_path), str(pairs_path))
                return
            except Exception as e:
                logging.warning('pycolmap verify_matches failed, fallback to CLI: %s', e)
        except Exception as e:
            logging.warning('pycolmap verify_matches failed, fallback to CLI: %s', e)

    cmd = [
        str(colmap_path), 'matches_importer',
        '--database_path', str(database_path),
        '--match_list_path', str(pairs_path),
        '--match_type', 'pairs',
        '--SiftMatching.use_gpu', '0',
    ]
    ret = subprocess.call(cmd, env=_colmap_headless_env())
    if ret != 0:
        logging.warning('Problem with matches_importer, existing.')
        exit(ret)


def create_db_from_model(empty_model, database_path):
    """ Create COLMAP database file from empty COLMAP binary file. """
    if database_path.exists():
        logging.warning('Database already exists.')
    
    cameras = read_cameras_binary(str(empty_model / 'cameras.bin'))
    images = read_images_binary(str(empty_model / 'images.bin'))

    db = COLMAPDatabase.connect(database_path)
    db.create_tables()
    
    for i, camera in cameras.items():
        model_id = CAMERA_MODEL_NAMES[camera.model].model_id
        db.add_camera(model_id, camera.width, camera.height, camera.params,
                      camera_id=i, prior_focal_length=True)
    
    for i, image in images.items():
        db.add_image(image.name, image.camera_id, image_id=i)
    
    db.commit()
    db.close()
    return {image.name: i for i, image in images.items()}


def import_features(image_ids, database_path, feature_path):
    """ Import keypoints info into COLMAP database. """
    logging.info("Importing features into the database...")
    feature_file = h5py.File(str(feature_path), 'r')
    db = COLMAPDatabase.connect(database_path)

    for image_name, image_id in tqdm.tqdm(image_ids.items()):
        keypoints = feature_file[image_name]['keypoints'].__array__()
        keypoints += 0.5
        db.add_keypoints(image_id, keypoints)
    
    feature_file.close()
    db.commit()
    db.close()


def import_matches(image_ids, database_path, pairs_path, matches_path, feature_path,
                   min_match_score=None, skip_geometric_verification=False):
    """ Import matches info into COLMAP database. """
    logging.info("Importing matches into the database...")

    with open(str(pairs_path), 'r') as f:
        pairs = [p.split(' ') for p in f.read().split('\n')]
    
    match_file = h5py.File(str(matches_path), 'r')
    db = COLMAPDatabase.connect(database_path)
    
    matched = set()
    for name0, name1 in tqdm.tqdm(pairs):
        id0, id1 = image_ids[name0], image_ids[name1]
        if len({(id0, id1), (id1, id0)} & matched) > 0:
            continue
    
        pair = names_to_pair(name0, name1)
        if pair not in match_file:
            raise ValueError(
                f'Could not find pair {(name0, name1)}... '
                'Maybe you matched with a different list of pairs? '
                f'Reverse in file: {names_to_pair(name0, name1) in match_file}.'
            )
        
        matches = match_file[pair]['matches0'].__array__()
        valid = matches > -1
        if min_match_score:
            scores = match_file[pair]['matching_scores0'].__array__()
            valid = valid & (scores > min_match_score)

        matches = np.stack([np.where(valid)[0], matches[valid]], -1)

        db.add_matches(id0, id1, matches)
        matched |= {(id0, id1), (id1, id0)}

        if skip_geometric_verification:
            db.add_two_view_geometry(id0, id1, matches)
    
    match_file.close()
    db.commit()
    db.close()


def run_triangulation(colmap_path, model_path, database_path, image_dir, empty_model):
    """ run triangulation on given database """
    logging.info('Running the triangulation...')
    # Prefer in-process pycolmap triangulation when available; fallback to CLI.
    pycolmap_ok = False
    try:
        import pycolmap  # type: ignore
    except Exception:
        pycolmap = None

    if pycolmap is not None and hasattr(pycolmap, "triangulate_points"):
        try:
            pycolmap.triangulate_points(
                database_path=str(database_path),
                image_path=str(image_dir),
                input_path=str(empty_model),
                output_path=str(model_path),
                options={
                    "Mapper.ba_refine_focal_length": False,
                    "Mapper.ba_refine_principal_point": False,
                    "Mapper.ba_refine_extra_params": False,
                },
            )
            pycolmap_ok = True
        except TypeError:
            # Backward compatibility for positional-only signatures.
            try:
                pycolmap.triangulate_points(
                    str(database_path),
                    str(image_dir),
                    str(empty_model),
                    str(model_path),
                )
                pycolmap_ok = True
            except Exception as e:
                logging.warning('pycolmap triangulate_points failed, fallback to CLI: %s', e)
        except Exception as e:
            logging.warning('pycolmap triangulate_points failed, fallback to CLI: %s', e)

    if not pycolmap_ok:
        cmd = [
            str(colmap_path), 'point_triangulator',
            '--database_path', str(database_path),
            '--image_path', str(image_dir),
            '--input_path', str(empty_model),
            '--output_path', str(model_path),
            '--Mapper.ba_refine_focal_length', '0',
            '--Mapper.ba_refine_principal_point', '0',
            '--Mapper.ba_refine_extra_params', '0'
        ]
        logging.info(' '.join(cmd))
        ret = subprocess.call(cmd, env=_colmap_headless_env())
        if ret != 0:
            logging.warning('Problem with point_triangulator, existing.')
            exit(ret)
    
    stats_raw = subprocess.check_output(
        [str(colmap_path), 'model_analyzer', '--path', model_path],
        env=_colmap_headless_env(),
    )
    stats_raw = stats_raw.decode().split('\n')
    stats = dict()
    for stat in stats_raw:
        if stat.startswith('Register images'):
            stats['num_reg_images'] = int(stat.split()[-1])
        elif stat.startswith('Points'):
            stats['num_sparse_points'] = int(stat.split()[-1])
        elif stat.startswith('Observation'):
            stats['num_observations'] = int(stat.split()[-1])
        elif stat.startswith('Mean track length'):
            stats['mean_track_length'] = float(stat.split()[-1])
        elif stat.startswith('Mean observation per image'):
            stats['num_observations_per_image'] = float(stat.split()[-1])
        elif stat.startswith('Mean reprojection error'):
            stats['mean_reproj_error'] = float(stat.split()[-1][:-2])
    return stats


def main(sfm_dir, empty_sfm_model, outputs_dir, pairs, features, matches, \
         colmap_path='colmap', skip_geometric_verification=False, min_match_score=None, image_dir=None):
    """ 
        Import keypoints, matches.
        Given keypoints and matches, reconstruct sparse model from given camera poses.
    """
    assert Path(empty_sfm_model).exists(), empty_sfm_model
    assert Path(features).exists(), features
    assert Path(pairs).exists(), pairs
    assert Path(matches).exists(), matches 

    Path(sfm_dir).mkdir(parents=True, exist_ok=True)
    database = osp.join(sfm_dir, 'database.db')
    model = osp.join(sfm_dir, 'model')
    Path(model).mkdir(exist_ok=True)

    image_ids = create_db_from_model(Path(empty_sfm_model), Path(database))
    import_features(image_ids, database, features)
    import_matches(image_ids, database, pairs, matches, features,
                   min_match_score, skip_geometric_verification)
    
    if not skip_geometric_verification:
        geometric_verification(colmap_path, database, pairs)
    
    if not image_dir:
        image_dir = '/'
    stats = run_triangulation(colmap_path, model, database, image_dir, empty_sfm_model)
    subprocess.call(
        [
            str(colmap_path), 'model_converter',
            '--input_path', str(model),
            '--output_path', str(Path(outputs_dir) / 'model.ply'),
            '--output_type', 'PLY',
        ],
        env=_colmap_headless_env(),
    )