import numpy as np
from easydict import EasyDict
import yaml
import torch
import argparse
import os
from scipy.spatial.transform import Rotation
from models.SIMECO import SIMECO

# Load configuration from a YAML file
def cfg_from_yaml_file(cfg_file):
    config = EasyDict()
    with open(cfg_file, 'r') as f:
        config.update(yaml.load(f, Loader=yaml.FullLoader))
    return config
 
def inference_single(root, pc_file, aug_mode = "no_agu"):
    # Initialize the PaCo model with the config      
    config = cfg_from_yaml_file(root + "/cfgs/SIMECO.yaml")
    checkpoint = torch.load(root + "/ckpt/checkpoint.pth")
    base_model = SIMECO(config.model)
   
    base_model.load_state_dict(checkpoint)
    base_model.to('cuda')
    base_model.eval()
    
    # Load the point cloud data
    n_points = 2048
    pc =  np.load(pc_file).astype(np.float32)
    choice = np.random.permutation(pc.shape[0])
    pc = pc[choice[:n_points]]
    if pc.shape[0] < n_points:
        zeros = np.zeros((n_points - pc.shape[0], 3))
        pc = np.concatenate([pc, zeros])

    # set random seed for reproducibility
    np.random.seed(42)
    if aug_mode == "rotation":
        # add rotation augmentation
        pc_file = pc_file.replace('/pc', '/pc_rotation')
        os.makedirs(os.path.dirname(pc_file), exist_ok=True)
        pc = np.matmul(pc, Rotation.random().as_matrix())
        np.save(pc_file, pc)
    elif aug_mode == "translation":
        pc_file = pc_file.replace('/pc', '/pc_translation')
        os.makedirs(os.path.dirname(pc_file), exist_ok=True)
        pc += np.random.uniform(-0.5, 0.5, size=(1, 3))
        np.save(pc_file, pc)
    elif aug_mode == "scale":
        pc_file = pc_file.replace('/pc', '/pc_scale')
        os.makedirs(os.path.dirname(pc_file), exist_ok=True)    
        # add random scale
        scale = np.random.uniform(0.5, 2.0)*0.5
        pc *= scale 
        np.save(pc_file, pc)
    elif aug_mode == "sim3":
        pc_file = pc_file.replace('/pc', '/pc_sim3')
        os.makedirs(os.path.dirname(pc_file), exist_ok=True)
        # add sim3 augmentation
        scale = np.random.uniform(0.5, 2.0)
        translation = np.random.uniform(-0.5, 0.5, size=(1, 3))
        rotation = Rotation.random().as_matrix()
        pc = np.matmul(pc, rotation) * scale + translation
        np.save(pc_file, pc)
         
    pc = torch.from_numpy(pc).float().unsqueeze(0).cuda() 
    ret = base_model(pc)
    dense_points = ret[-1].squeeze(0).detach().cpu().numpy()
    
    tgt_file = pc_file.replace('/pc', '/result')
    if not os.path.exists(os.path.dirname(tgt_file)):
        os.makedirs(os.path.dirname(tgt_file))
    
    np.save(tgt_file, dense_points)
    
if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description="Inference script")
    parser.add_argument('--root', type=str, default=current_directory, help='equiv directory path')
    parser.add_argument('--pc_file', type=str, required=True, help='incomplete point cloud file')
    parser.add_argument('--aug_mode', type=str, default="no_agu", help='aug mode')
    args = parser.parse_args()
    inference_single(args.root, args.pc_file, args.aug_mode)