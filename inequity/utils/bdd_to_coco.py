import argparse
import json
from os.path import join

import numpy as np
from PIL import Image
from tqdm import tqdm

BDD_CLASSES = ["bike",
               "bus",
               "car",
               "motor",
               "person",
               "rider",
               "traffic light",
               "traffic sign",
               "train",
               "truck"]


def make_coco_categories():
    """Make the cocostyle categories for
    BDD100K.

    Returns
    -------
    list

    """
    cats = []
    for i, bdd_class in enumerate(BDD_CLASSES):
        cat = {
            "supercategory": 'none',
            "id": i + 1,
            "name": bdd_class
        }
        cats.append(cat)
    return cats


def make_coco_images(src_json_file, img_dir, include, time_of_day):
    """Generate the cocostyle image metadata for BDD100K.

    Parameters
    ----------
        src_json_file : JSON
            BDD100K style det JSON file.
        img_dir : str
            Path to the BDD100K images directory.
        include : set of str
            Set of strs that represent which fitzpatrick
            scale categories to include.
        time_of_day : str
            Time of day to include. None if include all. 

    Returns
    -------
    (list, list)

    """
    file_names = []
    for i, anno in enumerate(src_json_file):
        if None not in include and anno["fitz_category"] not in include:
            continue
        if time_of_day is not None and anno["time_of_day"] != time_of_day:
            continue
        file_names.append(anno['name'])
    file_names = sorted(list(set(file_names)))

    imgs_metadata = []
    for i, file_name in enumerate(tqdm(file_names)):
        file_path = join(img_dir, file_name + ".jpg")
        width, height = Image.open(file_path).size
        img_metadata = {
            "height": height,
            "width": width,
            "id": i,
            "file_name": file_name + ".jpg"
        }
        imgs_metadata.append(img_metadata)
    return imgs_metadata, file_names


def make_coco_annotations(src_json_file,
                          include,
                          time_of_day,
                          ignore_occluded,
                          file_names):
    """Generate the cocostyle annotation metadata for BDD100K.

    Parameters
    ----------
        src_json_file : JSON
            BDD100K style det JSON file.
        include : set of str
            Set of strs that represent which fitzpatrick
            scale categories to include.
        time_of_day : str
            Time of day to include. None if include all. 
        ignore_occluded : bool
            Boolean flag whether to ignore occluded individuals.
        file_names : list
            List of included image files names.

    Returns
    -------
    list

    """

    anns = []
    num_included, num_ignored, num_occluded = 0, 0, 0
    for ann_id, ann in enumerate(src_json_file):
        category = ann['category']
        category = handle_category(ann['category'])

        bbox = ann['bbox']
        file_name = ann['name']
        if file_name not in file_names:
            continue
        bbox_width = bbox[2] - bbox[0] + 1
        bbox_height = bbox[3] - bbox[1] + 1
        xmin = bbox[0]
        ymin = bbox[1]
        box = [xmin, ymin, bbox_width, bbox_height]
        coco_ann = build_ann(ann,
                             file_name,
                             file_names,
                             box,
                             category,
                             ann_id)
        if ann["occluded"] \
                and ann['fitz_category'] in include \
                or None in include:
            num_occluded += 1
        if category == "person":
            if "fitz_category" in ann:
                fitz_category = ann["fitz_category"]
                if None not in include and fitz_category not in include \
                        or time_of_day is not None and ann["time_of_day"] != time_of_day:
                    coco_ann['ignore'] = 1
                    num_ignored += 1
                else:
                    num_included += 1
        anns.append(coco_ann)
    info = get_info(include, file_names, num_included,
                    num_ignored, num_occluded)
    print(info)
    if ignore_occluded:
        return set_ignore(anns, include)
    return anns


def build_ann(ann,
              img_id,
              img_ids,
              box,
              category,
              ann_id):
    """Build the cocostyle annotation from BDD100K data.

    Parameters
    ----------
        ann : dict
            BDD100K det style annotation
        img_id : str
            The unique image identifier for the image the
            annotation is in.
        img_ids : list
            A sorted list of img_ids for the entire dataset.
        box : list
            The bounding box in MS COCO coordinate format.
            (x_min, y_min, width, height)
        category : str
            The corresponding class the object belongs to.
        ann_id : int
            The unique identifier for the annotation.

    Returns
    -------
    dict

    """
    return {"image_id": img_ids.index(img_id),
            "bbox": box,
            "category_id": BDD_CLASSES.index(category) + 1,
            "segmentation": [[0, 0]],
            "area": box[-2] * box[-1],
            "id": ann_id,
            "iscrowd": 0,
            "fitz_category": ann['fitz_category'],
            "occluded": ann["occluded"],
            "weather": ann["weather"],
            "scene": ann["scene"],
            "time_of_day": ann["time_of_day"]}


def get_info(include, img_ids, num_included, num_ignored, num_occluded):
    """Get statistics about the current dataset.

    Parameters
    ----------
        include : set of str
            Set of strs that represent which fitzpatrick
            scale categories to include.
        img_ids : list
            A sorted list of img_ids for the entire dataset.
        num_included : int
            The number of person class included in the dataset.
        num_ignored : int
            The number of person class ignored in the dataset.
        num_occluded : int
            The number of occluded individuals in the dataset.

    Returns
    -------
    list

    """
    info = "{}\n".format(include)
    info += "Total images: {}\n".format(len(img_ids))
    info += "Total number of people: {}\n".format(num_included)
    info += "Total number of ignored: {}\n".format(num_ignored)
    info += "Total number of occluded: {}\n".format(num_occluded)
    return info


def handle_category(category):
    """Since MS COCO does not have a rider class, we
    change these categories to people in order to not
    unfairly penalize detection models using MS COCO
    weights.

    This does not affect largely affect BDD100K weights
    as they are trained with riders being correctly labeled,
    thus at test time they will likely just be predicted as
    the rider class (which will not be taken to account in person
    AP calculation).

    Parameters
    ----------
        category : str
            The category that the object belongs to.

    Returns
    -------
    str

    """
    return "person" if category == "rider" else category


def set_ignore(anns, include):
    """Ignore individuals in AP calculation.

    Parameters
    ----------
        anns : list
            The list of annotations in the dataset.
        include : set of str
            Set of strs that represent which fitzpatrick
            scale categories to include.

    Returns
    -------
    list

    """
    ignore_count = 0
    for ann in anns:
        is_person = ann["category_id"] == 5
        is_occluded = ann["occluded"]
        should_include = ann["fitz_category"] in include
        if is_person and is_occluded and should_include:
            ann["ignore"] = 1
            ignore_count += 1
    info = "Total number of occluded ignored: {}\n".format(ignore_count)
    print(info)
    return anns


def bdd_to_coco(src_path,
                dst_path,
                img_dir,
                include,
                ignore_occluded=False,
                time_of_day=None):
    """Convert BDD100K det format to MS COCO format.

    Parameters
    ----------
        src_path : str
            The path to the BDD100K det JSON file.
        dst_path : str
            The destination path to save the MS COCO
            annotations.
        img_dir : str
            Path to the BDD100K images directory.
        include : str
            Which category to include.
        time_of_day : str
            Time of day to include. None if include all. 
        ignore_occluded : bool
            Boolean flag whether to ignore occluded individuals.

    Returns
    -------
    list

    """
    src = json.load(open(src_path))
    include = set([include])

    bdd100k_cocostyle = {}
    bdd100k_cocostyle['categories'] = make_coco_categories()
    bdd100k_cocostyle['images'], filenames = make_coco_images(
        src, img_dir, include, time_of_day)
    bdd100k_cocostyle['annotations'] = make_coco_annotations(
        src, include, time_of_day, ignore_occluded, filenames)

    with open(dst_path, 'w') as json_file:
        json.dump(bdd100k_cocostyle, json_file)
