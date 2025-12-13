import hydra
from omegaconf import DictConfig
import numpy as np
import os
from scipy.spatial.transform import Rotation

import torch

from utils.logger import get_root_logger
from utils import builder

def inference_single(cfg):
    """
    Perform inference on a single point cloud file using the PaCo model.
    
    Args:
        cfg: Configuration object containing model and evaluation parameters
    """
    # Initialize the PaCo model with the config      
    base_model = builder.model_builder(cfg.model)
    builder.load_model(base_model, cfg.evaluate.checkpoint_path)
   
    # Move model to GPU and set to evaluation mode
    base_model.to('cuda')
    base_model.eval()
    
    # Load and preprocess the point cloud data
    n_points = 2048  # Target number of points for the model
    pc_file = cfg.evaluate.single_file_path
    pc = np.load(pc_file).astype(np.float32)
    
    # Randomly sample points to match target size
    choice = np.random.permutation(pc.shape[0])
    pc = pc[choice[:n_points]]
    
    # Pad with zeros if we have fewer points than needed
    if pc.shape[0] < n_points:
        zeros = np.zeros((n_points - pc.shape[0], 3))
        pc = np.concatenate([pc, zeros])

    # Set random seed for reproducibility of augmentations
    np.random.seed(42)
    
    # Apply data augmentation based on the specified mode
    if cfg.evaluate.aug_mode == "rotation":
        # Apply random rotation augmentation
        pc_file = pc_file.replace('/pc', '/pc_rotation')
        os.makedirs(os.path.dirname(pc_file), exist_ok=True)
        pc = np.matmul(pc, Rotation.random().as_matrix())
        np.save(pc_file, pc)
    elif cfg.evaluate.aug_mode == "translation":
        # Apply random translation augmentation
        pc_file = pc_file.replace('/pc', '/pc_translation')
        os.makedirs(os.path.dirname(pc_file), exist_ok=True)
        pc += np.random.uniform(-0.5, 0.5, size=(1, 3))
        np.save(pc_file, pc)
    elif cfg.evaluate.aug_mode == "scale":
        # Apply random scale augmentation
        pc_file = pc_file.replace('/pc', '/pc_scale')
        os.makedirs(os.path.dirname(pc_file), exist_ok=True)    
        scale = np.random.uniform(0.5, 2.0) * 0.5
        pc *= scale 
        np.save(pc_file, pc)
    elif cfg.evaluate.aug_mode == "sim3":
        # Apply similarity transformation (rotation + scale + translation)
        pc_file = pc_file.replace('/pc', '/pc_sim3')
        os.makedirs(os.path.dirname(pc_file), exist_ok=True)
        scale = np.random.uniform(0.5, 2.0)
        translation = np.random.uniform(-0.5, 0.5, size=(1, 3))
        rotation = Rotation.random().as_matrix()
        pc = np.matmul(pc, rotation) * scale + translation
        np.save(pc_file, pc)
         
    # Convert to PyTorch tensor and move to GPU
    pc = torch.from_numpy(pc).float().unsqueeze(0).cuda() 
    
    # Run inference through the model
    ret = base_model(pc)
    dense_points = ret[-1].squeeze(0).detach().cpu().numpy()
    
    # Save the reconstructed dense point cloud
    tgt_file = pc_file.replace('/pc', '/result')
    if not os.path.exists(os.path.dirname(tgt_file)):
        os.makedirs(os.path.dirname(tgt_file))
    
    np.save(tgt_file, dense_points)
   
    
@hydra.main(config_path='./conf', config_name='config', version_base='1.2')
def inference(cfg: DictConfig):
    """
    Main inference function that sets up the environment and runs PaCo reconstruction.
    
    Args:
        cfg: Hydra configuration object containing all test parameters
    """
    # Set up logger
    logger = get_root_logger(name=cfg.log_name)
    
    # Configure GPU settings if available
    if cfg.use_gpu and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        assert not cfg.distributed, "Distributed testing is not supported."

    # Validate that input file path is provided
    assert cfg.evaluate.single_file_path is not None 
    
    # Execute the reconstruction process
    logger.info("Starting reconstruction...")
    inference_single(cfg)


if __name__ == "__main__":
    inference()
