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
    data_type = data["_object_type"]
    if data_type == "sampling":
        from ASFEniCSx.sampling import Sampling
        object = Sampling(data["M"], data["m"])
        object.load(data, overwrite=True)
    elif data_type=="clustering":
        from ASFEniCSx.sampling import Clustering
        object = Clustering(data["M"], data["m"], data["k"], data["_max_iter"])
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
               interval: np.ndarray = np.array([-1., 1.]),
               debug : bool = False):
    """Normalizes a sample
    
    Normalizes a sample to the interval [-1,1] using the bounds
    
    Args:
        sample (np.ndarray): The sample to be normalized. Has either shape (M, m) or (m,).
        bounds (np.ndarray): The bounds used for normalization. Has shape (m, 2).
        interval (np.ndarray): The interval for normalization, Default [-1., 1.]
    Returns:
        np.ndarray: The normalized sample
    """

    debug_info(debug, f"Normalizing sample {np.shape(sample)} with bounds {np.shape(bounds)} and interval {np.shape(interval)}")

    # Make sample 2D if it is 1D
    if len(sample.shape) == 1:
        sample = sample.reshape(1, -1)
    
    # Check if bounds are valid
    if bounds.shape[0] != sample.shape[1] or bounds.shape[1]!= 2:
        raise ValueError(f"Bounds do not match sample dimensions. Have shape {np.shape(bounds)} but should have shape ({sample.shape[1]}, 2)")
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("Lower bounds seem to be greater than Upper bounds")
    
    # Normalize
    sample = \
        (interval[1] - interval[0]) * (sample - bounds[:, 0]) / \
        (bounds[:, 1] - bounds[:, 0]) + interval[0]

    # Make sample 1D if it is unnecessary 2D
    if sample.shape[0] == 1:
        sample = sample.reshape(-1)
    return sample

def denormalizer(sample: np.ndarray, bounds: np.ndarray,
                 interval = np.array([-1., 1.]),
                 debug : bool = False):
    """Denormalizes a sample
    
    Denormalizes a sample from the interval using the bounds
    
    Args:
        sample (np.ndarray): The sample to be unnormalized. Has either shape (M, m) or (m, ).
        bounds (np.ndarray): The bounds used for unnormalization. Has shape (m, 2). 
        interval (np.ndarray): Scaling interval. Default [-1., 1.]
    Returns:
        np.ndarray: The unnormalized sample
    """

    debug_info(debug, f"Denormalizing sample {np.shape(sample)} with bounds {np.shape(bounds)} and interval {np.shape(interval)}")


    # Make sample 2D if it is 1D
    if len(sample.shape) == 1:
        sample = sample.reshape(1, -1)

    # Check if bounds are valid
    if bounds.shape[0] != sample.shape[1]:
        raise ValueError("Bounds do not match sample dimensions")
    
    # Check if bounds are valid
    if np.any(bounds[:, 0] > bounds[:, 1]):
        raise ValueError("Lower bounds seem to be greater than Upper bounds")
    
    # Unnormalize
    sample = (sample - interval[0]) * (bounds[:, 1] - bounds[:, 0]) / \
        (interval[1] - interval[0]) + bounds[: , 0]

    # Make sample 1D if it is unnecessary 2D
    if sample.shape[0] == 1:
        sample = sample.reshape(-1)

    return sample