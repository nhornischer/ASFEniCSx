import json
import numpy as np
from inspect import currentframe, getframeinfo

class NumpyEncoder(json.JSONEncoder):
    """Class for encoding numpy arrays to json
    
    This class is used to encode numpy arrays to json. It is used in the save method of the sampling and clustering classes.
    
    Methods:
    public:
        default(obj) -> json: Encodes the given object to json
        
    Example:
        >>> json.dumps(np.array([1,2,3]), cls=NumpyEncoder)
        '[1, 2, 3]'
    
    Version:
        0.1
    Contributors:
        Niklas Hornischer (nh605@cam.ac.uk)
    """

    def default(self, obj : object):
        """Returns a converted object that can be converted to json
        
        Args:
            obj (object): The object to be encoded
            
        Returns:
            (object) object that can be converted to json
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()


def load(filename : str):
    """Loads a sampling object from a json file
    
    Args:
        filename (str): The name of the file to be loaded
        
    Returns:
        object: The sampling object
    
    Raises:
        FileNotFoundError: If the file does not exist
    """
    try:
        with open(filename, "r") as f:
            data=json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError("File not found")
    data_type = data["object_type"]
    if data_type == "sampling":
        from ASFEniCSx.sampling import sampling
        object = sampling(data["M"], data["m"])
        object.load(data, overwrite=True)
    elif data_type=="clustering":
        from ASFEniCSx.sampling import clustering
        object = clustering(data["M"], data["m"], data["k"], data["_max_iter"])
        object.load(data)
    
    return object

def debug_info(debug : bool, message : str):
    """Prints debug information
    
    Prints the given message if the debug flag is set to True
    
    Args:
        message (str): The message to be printed
    """
    if debug:
        frameinfo = getframeinfo(currentframe().f_back)
        print(f"DEBUG: File \"{frameinfo.filename}\", line {frameinfo.lineno}, module {frameinfo.function} \n\t{message}")    

def normalizer(sample : np.ndarray, bounds : np.ndarray,
               interval: np.ndarray = np.array([-1., 1.])):
    """Normalizes a sample
    
    Normalizes a sample to the interval [-1,1] using the bounds
    
    Args:
        sample (np.ndarray): The sample to be normalized. Has either shape (M, m) or (m,).
        bounds (np.ndarray): The bounds used for normalization. Has shape (2, m). # NOTE bounds shape changed from (m, 2) to (2, m)
        interval (np.ndarray): The interval for normalization, Default [-1., 1.] # NOTE
    Returns:
        np.ndarray: The normalized sample
    """

    # Make sample 2D if it is 1D
    if len(sample.shape) == 1:
        sample = sample.reshape(1, -1)
    
    # Check if bounds are valid
    if bounds.shape[1] != sample.shape[1]:
        raise ValueError("Bounds do not match sample dimensions")
    
    # Check if bounds are valid
    if np.any(bounds[0, :] > bounds[1, :]):
        raise ValueError("Lower bounds seem to be greater than Upper bounds")
    
    '''
    TODO Is this necessary?
    # Check if sample is valid
    if np.any(sample < bounds[:,0]) or np.any(sample > bounds[:,1]):
        raise ValueError("Sample is invalid")
    '''
    
    # Normalize
    sample = \
        (interval[1] - interval[0]) * (sample - bounds[0, :]) / \
        (bounds[1, :] - bounds[0, :]) + interval[0]

    # Make sample 1D if it is unnecessary 2D
    if sample.shape[0] == 1:
        sample = sample.reshape(-1)
    return sample

def denormalizer(sample: np.ndarray, bounds: np.ndarray,
                 interval=np.array([-1., 1.])):
    """Denormalizes a sample
    
    Denormalizes a sample from the interval using the bounds
    
    Args:
        sample (np.ndarray): The sample to be unnormalized. Has either shape (M, m) or (m, ).
        bounds (np.ndarray): The bounds used for unnormalization. Has shape (2, m). # NOTE bounds shape changed from (m, 2) to (2, m)
        interval (np.ndarray): Scaling interval. Default [-1., 1.] # NOTE
    Returns:
        np.ndarray: The unnormalized sample
    """

    # Make sample 2D if it is 1D
    if len(sample.shape) == 1:
        sample = sample.reshape(1, -1)

    # Check if bounds are valid
    if bounds.shape[1] != sample.shape[1]:
        raise ValueError("Bounds do not match sample dimensions")
    
    # Check if bounds are valid
    if np.any(bounds[0, :] > bounds[1, :]):
        raise ValueError("Lower bounds seem to be greater than Upper bounds")
    
    '''
    TODO Is this necessary?
    # Check if sample is valid
    if np.any(sample < -1) or np.any(sample > 1):
        raise ValueError("Sample is invalid")
    '''
    
    # Unnormalize
    sample = (sample - interval[0]) * (bounds[1, :] - bounds[0, :]) / \
        (interval[1] - interval[0]) + bounds[0, :]

    # Make sample 1D if it is unnecessary 2D
    if sample.shape[0] == 1:
        sample = sample.reshape(-1)

    return sample


if __name__ == "__main__":
    a_original = np.random.uniform(-1.3, 2., (10, 2))
    bounds = np.vstack([np.array([-1.3, -1.3]), np.array([2., 2.])])
    interval = np.array([-0.5, 0.5])
    normalized_a = normalizer(a_original, bounds, interval)
    denormalized_a = denormalizer(normalized_a, bounds, interval)
    print(f"Original a: {a_original}")
    print(f"Normalized a: {normalized_a}")
    print(f"Denormalized a: {denormalized_a}")
    print(f"Max error between original and denormalized sample: {np.max(np.abs(a_original - denormalized_a))}")
    
    '''
    TODO
    1. Is checking sample validity necessary?
    NOTE
    1. Bounds shape changed from (m, 2) to (2, m) for consistencny with sample shape
    2. Interval range [-1., 1.] is default but not mandatory
    '''
