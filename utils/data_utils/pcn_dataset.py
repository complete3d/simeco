import json
import os
import random
import sys

import numpy as np
import torch
import torch.utils.data as data

# Set up the base directory and add it to the system path for local imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from utils.logger import *
from .build import DATASETS
from .io import IO
import data_transforms

from scipy.spatial.transform import Rotation


@DATASETS.register_module()
class PCN(data.Dataset):
    """
    Dataset class for the PCN dataset.
    
    This dataset loads complete and partial point clouds.
    It supports training and testing subsets and applies a sequence of transformations
    to the data.
    """
    
    def __init__(self, config):
        """
        Initialize the PCN dataset with configuration parameters.
        
        Args:
            config: Configuration object containing dataset paths and parameters
        """
        # Set dataset paths and parameters from config
        self.partial_points_path = config.partial_points_path
        self.complete_points_path = config.complete_points_path
        self.category_file = config.category_file_path
        self.num_points = config.num_points
        self.subset = config.subset
        self.cars = config.cars
        self.mode = config.mode
        self.random_object_scale = config.random_object_scale

        # Load the dataset indexing file containing category information
        self.dataset_categories = []
        with open(self.category_file) as f:
            self.dataset_categories = json.loads(f.read())
            # Filter to only car category if specified
            if self.cars:
                self.dataset_categories = [
                    dc for dc in self.dataset_categories 
                    if dc['taxonomy_id'] == '02958343'
                ]

        # Set number of renderings based on subset (more for training)
        self.n_renderings = config.num_renderings if self.subset == 'train' else 1
        
        # Generate file list and set up transforms
        self.file_list = self._get_file_list(self.subset, self.n_renderings)
        self.transforms = self._get_transforms(self.subset)

    def _get_transforms(self, subset):
        """Set up data transforms based on dataset subset.
        
        Args:
            subset: Dataset subset ('train', 'test', etc.)
            
        Returns:
            Composed transforms for data preprocessing
        """
        if subset == 'train':
            return data_transforms.Compose([
                {
                    'callback': 'RandomSamplePoints',
                    'parameters': {'n_points': 2048},
                    'objects': ['partial']
                },
                {
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt']
                }
            ])
        else:
            return data_transforms.Compose([
                {
                    'callback': 'RandomSamplePoints',
                    'parameters': {'n_points': 2048},
                    'objects': ['partial']
                },
                {
                    'callback': 'ToTensor',
                    'objects': ['partial', 'gt']
                }
            ])

    def _sim3_transform(self, data):
        """Apply Sim3 transformation to the point clouds.
        
        Args:
            data: Dictionary containing 'partial' and 'gt' point clouds
            
        Returns:
            Transformed data dictionary with applied Sim3 transformation
        """
        pc, gt = data['partial'], data['gt']
        centroid = data['centroid']
        pc = pc + centroid
        gt = gt + centroid
        
        if not self.random_object_scale:
            scale = 1.0 / torch.max(torch.norm(pc, dim=1))
            data['scale'] = torch.from_numpy(scale.numpy()).type(pc.dtype)
        else:  
            scale = (torch.rand(1) * (self.random_object_scale[1] - self.random_object_scale[0]) + self.random_object_scale[0])
            data['scale'] = torch.from_numpy(scale.numpy()).type(pc.dtype)
        
        pc = pc * scale
        gt = gt * scale
        
        random_R = torch.from_numpy(Rotation.random().as_matrix()).type(pc.dtype) 
        data['rotation'] = random_R
        data['partial'] = torch.matmul(pc, random_R)
        data['gt'] = torch.matmul(gt, random_R) 
        
        data['centroid'] = torch.matmul(data['centroid'], random_R)
        
        return data

    def _get_file_list(self, subset, n_renderings=1):
        """Prepare file list for the dataset by collecting all sample paths.
        
        Args:
            subset: Dataset subset to collect files for
            n_renderings: Number of rendering views per sample
            
        Returns:
            List of dictionaries containing file paths and metadata
        """
        file_list = []

        # Iterate through each category in the dataset
        for dc in self.dataset_categories:
            print_log(
                f'Collecting files of Taxonomy [ID={dc["taxonomy_id"]}, Name={dc["taxonomy_name"]}]',
                logger='PCNDATASET'
            )
            samples = dc[subset]

            # Process each sample in the category
            for s in samples:
                file_list.append({
                    'taxonomy_id': dc['taxonomy_id'],
                    'model_id': s,
                    'partial_path': [
                        self.partial_points_path % (subset, dc['taxonomy_id'], s, i)
                        for i in range(n_renderings)
                    ],
                    'gt_path': self.complete_points_path % (subset, dc['taxonomy_id'], s),
                })

        print_log(
            f'Complete collecting files of the dataset. Total files: {len(file_list)}',
            logger='PCNDATASET'
        )
        return file_list

    def __getitem__(self, idx):
        """Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple containing (taxonomy_id, model_id, (partial_points, complete_points))
        """
        sample = self.file_list[idx]
        data = {}
        
        # Select random rendering for training, first rendering for other subsets
        rand_idx = random.randint(0, self.n_renderings - 1) if self.subset == 'train' else 0

        # Load partial and ground truth point clouds
        for ri in ['partial', 'gt']:
            file_path = sample[f'{ri}_path']
            
            # Handle multiple renderings (select one based on rand_idx)
            if isinstance(file_path, list):
                file_path = file_path[rand_idx]
                
            data[ri] = IO.get(file_path).astype(np.float32)
            
            if self.mode == 'sim3' and self.subset == 'test':
                pc_centroid = np.mean(data['partial'], axis=0)
                pc_scale = 1/np.max(np.linalg.norm(data['partial'] - pc_centroid, axis=1))
                data['centroid'] = -torch.tensor(pc_centroid).float()
            else:
                data['centroid'] = torch.zeros(3).float()
                data['scale'] = torch.ones(1).float()
                

        # Ensure ground truth has correct number of points
        assert data['gt'].shape[0] == self.num_points

        # Apply data transforms if specified
        if self.transforms is not None:
            data = self.transforms(data)
            
        if self.mode == 'sim3' and self.subset == 'test':
            data = self._sim3_transform(data)
            return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'], data['scale'])
          
        return sample['taxonomy_id'], sample['model_id'], (data['partial'], data['gt'])

    def __len__(self):
        """Return the total number of samples in the dataset.
        
        Returns:
            Integer representing dataset size
        """
        return len(self.file_list)
