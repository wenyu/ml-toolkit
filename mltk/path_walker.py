import os

TYPE_IMAGE = [".jpg", ".jpeg", ".png"]
TYPE_XML = [".xml"]
TYPE_JSON = [".json"]


def get_files_within_directory(path, type=None):
    result = []
    try:
        path = os.path.abspath(path)
        for f in os.listdir(path):
            p = os.path.join(path, f)
            if os.path.isdir(p):
                result += get_files_within_directory(p, type)
                continue

            elif type is not None:
                ext = os.path.splitext(p)[1].lower()
                if ext not in type:
                    continue
            result.append(p)
    except:
        pass

    return result