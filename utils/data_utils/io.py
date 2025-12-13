import numpy as np
import open3d
import os


class IO:
    """
    A utility class for reading various file formats containing point cloud or numeric data.
    Supports .npy, .pcd, .ply, and .txt file extensions.
    """
    
    @classmethod
    def get(cls, file_path):
        """
        Read data from a file based on its extension.
        
        Args:
            file_path (str): Path to the file to be read
            
        Returns:
            np.ndarray: The loaded data as a numpy array
            
        Raises:
            Exception: If the file extension is not supported
        """
        # Extract file extension from the file path
        _, file_extension = os.path.splitext(file_path)

        # Route to appropriate reader based on file extension
        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        elif file_extension in ['.pcd', '.ply']:
            return cls._read_pcd(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    @classmethod
    def _read_npy(cls, file_path):
        """
        Read numpy array from .npy file.
        
        Args:
            file_path (str): Path to the .npy file
            
        Returns:
            np.ndarray: Loaded numpy array
        """
        return np.load(file_path)
       
    @classmethod
    def _read_pcd(cls, file_path):
        """
        Read point cloud from .pcd or .ply file using Open3D.
        
        Args:
            file_path (str): Path to the point cloud file
            
        Returns:
            np.ndarray: Point cloud data as numpy array of shape (N, 3)
        """
        # Load point cloud using Open3D
        pc = open3d.io.read_point_cloud(file_path)
        # Convert to numpy array containing only the points
        ptcloud = np.array(pc.points)
        return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        """
        Read numeric data from .txt file.
        
        Args:
            file_path (str): Path to the text file
            
        Returns:
            np.ndarray: Loaded data as numpy array
        """
        return np.loadtxt(file_path)
