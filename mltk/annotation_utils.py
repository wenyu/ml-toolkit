import json
import operator as op
import os
import xml.etree.ElementTree as ET

def pascal_voc_to_objects(pascal_voc_obj):
    """
    Convert a XML ElementTree object in PascalVOC format to a format easy for YOLO to use.

    :type pascal_voc_obj: xml.etree.ElementTree.ElementTree
    :return list[dict]:
    """
    def object_to_dict(obj):
        name = obj.findtext("name")
        box = obj.find("bndbox")
        xmin = int(box.findtext("xmin"))
        xmax = int(box.findtext("xmax"))
        ymin = int(box.findtext("ymin"))
        ymax = int(box.findtext("ymax"))
        return {
            "label": name,
            "coordinates": {
                "xmin": xmin,
                "xmax": xmax,
                "ymin": ymin,
                "ymax": ymax,
                # x, y are center point of box.
                "x": (xmin + xmax) / 2,
                "y": (ymin + ymax) / 2,
                "width": xmax - xmin,
                "height": ymax - ymin,
            }
        }
    
    return map(object_to_dict, pascal_voc_obj.findall("object"))


def object_coordinate_to_bounding_boxes(objs):
    """
    Convert to imgaug BoundingBox.

    :type objs: dict | xml.etree.ElementTree.ElementTree
    :return: imgaug.imgaug.BoundingBox
    """
    if type(objs) != "dict":
        objs = pascal_voc_to_objects(objs)
    objs = map(op.itemgetter("coordinates"), objs)
    import imgaug as ia
    return map(lambda o: ia.BoundingBox(x1=o["xmin"], y1=o["ymin"], x2=o["xmax"], y2=o["ymax"]), objs)


def bounding_box_to_object_coordinate(bbox):
    """
    Convert an imgaug BoundingBox back to coordinate

    :type bbox: imgaug.imgaug.BoundingBox
    :return: dict
    """
    xmin, xmax, ymin, ymax = bbox.x1, bbox.x2, bbox.y1, bbox.y2
    return {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        # x, y are center point of box.
        "x": (xmin + xmax) / 2,
        "y": (ymin + ymax) / 2,
        "width": xmax - xmin,
        "height": ymax - ymin,
    }


def guess_annotation_path(path, annotation_search_paths=[], annotation_ext=[".json", ".xml"]):
    """
    Try to locate and load the annotation for given image.

    :type path: str
    :return: str | None
    """
    path = os.path.abspath(path)
    base_path, fn = os.path.split(path)
    base_fn, ext = os.path.splitext(fn)
    annotation_search_paths.append(base_path)

    for annotation_dir in annotation_search_paths:
        fn = os.path.join(annotation_dir, base_fn)
        for ext in annotation_ext:
            result = fn + ext
            if os.path.exists(result):
                return result

    return None


def load_object_coordinate_annotation(path, annotation_search_paths=[]):
    """
    Load annotation from given file.

    :type path: str
    :return: list[dict]
    """
    try:
        _, ext = os.path.splitext(path)
        if ext not in [".json", ".xml"]:
            path = guess_annotation_path(path, annotation_search_paths)
        _, ext = os.path.splitext(path)
        ext = ext.lower()
        if ext == ".json":
            with open(path, "r") as f:
                return json.load(f)
        elif ext == ".xml":
            return pascal_voc_to_objects(ET.parse(path))
    except:
        return []