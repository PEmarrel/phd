import json


def DicToJson(dic: dict, path: str):
    """
    Save dict to json file, attempting to convert numeric strings to numbers.
    """
    # Create a copy or transform the data to avoid string-numbers
    cleaned_dic = {k: try_cast(v) for k, v in dic.items()}
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned_dic, f, ensure_ascii=False, indent=4)

def try_cast(value):
    """Helper to convert string numbers to int or float."""
    if not isinstance(value, str):
        return value
    try:
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return int(value)
        return float(value)
    except ValueError:
        return value


def parse_numbers(d):
    for k, v in d.items():
        if isinstance(v, str):
            # Try to convert to int
            try:
                d[k] = int(v)
            except ValueError:
                # Try to convert to float
                try:
                    d[k] = float(v)
                except ValueError:
                    pass # Keep as string if it's not a number
    return d

def JsonToDic(path: str) -> dict:
    """
    Load a json file into a dict
    
    Args :
        path : str
    
    Returns :
        dict
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f, object_hook=parse_numbers)