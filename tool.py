import json


def DicToJson(dic: dict, path: str):
    """
    Save dict to json file
    
    Args :
        dic : dict
        path : str
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dic, f, ensure_ascii=False, indent=4)


def JsonToDic(path: str) -> dict:
    """
    Load a json file into a dict
    
    Args :
        path : str
    
    Returns :
        dict
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
