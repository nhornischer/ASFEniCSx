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
