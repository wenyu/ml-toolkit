import operator as op

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


def get_imgaug_bounding_boxes(objs):
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
