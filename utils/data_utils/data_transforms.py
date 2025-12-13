import numpy as np
import torch

class Compose(object):
    """
    Composes several transformations together.

    Each transformation is defined by a dictionary that specifies a 'callback' (the name of the
    transformation class as a string), optional 'parameters' for the transformation, and 'objects'
    which lists the data keys to which the transformation should be applied.
    """
    def __init__(self, transforms):
        """
        Initialize the Compose object with a list of transformation dictionaries.

        Args:
            transforms: List of dictionaries. Each dictionary contains:
                - 'callback': Name of the transformation class (as a string)
                - 'parameters': (Optional) Parameters for the transformation
                - 'objects': List of keys in the data to apply this transformation to
        """
        self.transformers = []
        for tr in transforms:
            # Dynamically retrieve the transformation class using eval
            transformer = eval(tr['callback'])
            parameters = tr['parameters'] if 'parameters' in tr else None
            self.transformers.append({
                'callback': transformer(parameters),
                'objects': tr['objects']
            })  # yapf: disable

    def __call__(self, data):
        """
        Apply the composed transformations to the input data.

        Args:
            data: Dictionary containing the data to be transformed

        Returns:
            The transformed data dictionary
        """
        for tr in self.transformers:
            transform = tr['callback']
            objects = tr['objects']
            # Generate a random value (currently not used)
            rnd_value = np.random.uniform(0, 1)
            
            for k, v in data.items():
                if k in objects and k in data:
                    data[k] = transform(v)                
        return data

class ToTensor(object):
    """
    Convert a numpy array to a PyTorch tensor.
    """
    def __init__(self, parameters):
        """
        Initialize the ToTensor transformation.

        Args:
            parameters: Not used for this transformation
        """
        pass

    def __call__(self, arr):
        """
        Convert a numpy array to a PyTorch tensor of type float.

        Args:
            arr: Numpy array to convert

        Returns:
            A PyTorch tensor
        """
        return torch.from_numpy(arr.copy()).float()


class RandomSamplePoints(object):
    """
    Randomly sample a fixed number of points from a point cloud.
    """
    def __init__(self, parameters):
        """
        Initialize the RandomSamplePoints transformation.
        
        Args:
            parameters: Dictionary containing 'n_points', the number of points to sample
        """
        self.n_points = parameters['n_points']

    def __call__(self, ptcloud):
        """
        Randomly sample points from the input point cloud.
        
        Args:
            ptcloud: Numpy array of shape (N, 3) representing the point cloud
            
        Returns:
            Numpy array of shape (n_points, 3) representing the sampled point cloud
        """
        choice = np.random.permutation(ptcloud.shape[0])
        ptcloud = ptcloud[choice[:self.n_points]]

        if ptcloud.shape[0] < self.n_points:
            zeros = np.zeros((self.n_points - ptcloud.shape[0], 3))
            ptcloud = np.concatenate([ptcloud, zeros])

        return ptcloud