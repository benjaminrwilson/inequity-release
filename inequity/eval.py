import json
import os
import shutil
from os import listdir
from os.path import expanduser, join

import cv2
import numpy as np
import torch
from tabulate import tabulate
from tqdm import tqdm

from demo.predictor import *
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from utils.bdd_to_coco import bdd_to_coco
from utils.utils import *


def compute_metrics(coco_gt, results):
    """Calculates COCO metrics for a set of ground
    truth annotations and predicted bounding boxes/classes.

    Parameters
    ----------
    coco_gt : COCO
        Instance of COCO object loaded with an annotation file.
    results : ndarray
        The results from the detection model in COCO results format.

    Returns
    -------
    ndarray

    """
    disable_print()
    coco_dt = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.catIds = [5]
    coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2],
                                [0 ** 2, 15000],
                                [15000, 25000],
                                [25000, 1e5 ** 2]]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    enable_print()
    print()
    return coco_eval.stats


def eval_model(home,
               img_dir,
               json_path,
               model_name,
               mode,
               weights_path=None):
    """Evaluates a model with a specified weights file
    over all the images in img_dir.

    Parameters
    ----------
    home : str
        Home directory in terminal.
    img_dir : str
        Directory where BDD100K images are stored (for a particular split).
    json_path : str
        Path to JSON annotation file.
    model_name : str
        Model name used for evaluation.
    mode : EvalMode (Enum)
        Mode for evaluation.
    weights_path : str
        The path to the weights file for evaluation.

    Returns
    -------
    ndarray

    """
    model = get_model(home, model_name, weights_path=weights_path, mode=mode)

    disable_print()
    coco_gt = COCO(json_path)
    enable_print()

    image_ids = set([img["file_name"] for img in coco_gt.imgs.values()])

    results = []
    dst_img_dir = join(home, "inequity-release", "images")
    shutil.rmtree(dst_img_dir, ignore_errors=True)
    os.makedirs(dst_img_dir)
    for i, img_name in enumerate(tqdm(sorted(image_ids))):
        abs_img_path = join(img_dir, img_name)
        img = cv2.imread(abs_img_path)
        predictions = model.compute_prediction(img)

        draw_predictions(img, img_name, predictions, coco_gt, i)

        detections = prepare_detections(i, predictions, mode)
        results += detections
    results = np.array(results)
    return compute_metrics(coco_gt, results)


def run_trials(home,
               root,
               img_dir,
               json_paths,
               model_names,
               mode,
               trials=1,
               weight_dirs=[""]):
    """Run every combination of input
    parameters using MS COCO tools.

    Parameters
    ----------
    home : str
        Home directory in terminal.
    root : str
        Root directory to where weights folders are stored.
    img_dir : str
        Directory where BDD100K images are stored (for a particular split).
    json_paths : list of str
        List of paths to JSON annotation files.
    model_names : list of str
        List of model names used for evaluation.
    mode : EvalMode (Enum)
        Mode for evaluation.
    trials : int
        The number of different weights to use from training.
    weight_dirs : list of str
        A list of weight directories for evaluation.

    Returns
    -------
    ndarray

    """
    if mode == EvalMode.COCO:
        weight_dirs = ["MS COCO"]

    n = len(weight_dirs)
    z = len(model_names)
    m = 1 if mode != EvalMode.ALL else len(
        get_weight_paths(root, weight_dirs[0], 0, mode))

    results = np.zeros((n, z, trials, m, 2, 12))

    for i, weight_dir in enumerate(weight_dirs):
        for j, model_name in enumerate(model_names):
            print("Model Name: {}".format(model_name))
            for k in range(trials):
                print("Beginning trial {} ...".format(k))
                weight_paths = get_weight_paths(root, weight_dir, k, mode)
                for l, weight_path in enumerate(weight_paths):
                    weight_info = "MS COCO" if mode == EvalMode.COCO else weight_path
                    print("Weights: {}".format(weight_info))
                    for p, json_path in enumerate(json_paths):
                        json_info = json_path.split("/")[-1]
                        print("Json Info: {}".format(json_info))
                        results[i, j, k, l, p, :] = eval_model(home,
                                                               img_dir,
                                                               json_path,
                                                               model_name,
                                                               mode,
                                                               weight_path)
    return results


def run(home, gen_flags, split, subsets):
    """Generate selected tables.

    Parameters
    ----------
    home : str
        Home directory in terminal.
    include_tables : dict
        Determine which tables are calculated.
    split : str
        Split to evaluate on. Choose either "train" or "val".
    subsets : list of str
        Subsets to evaluate on.

    Returns
    -------
    None

    """

    root = join(home, "inequity-release", "weights")
    img_dir = join(home, "inequity-release", "datasets/bdd100k/images/100k/{}".format(split))

    ann_root = join(home, "inequity-release", "datasets/annotations")
    bdd_src = join(ann_root, "bdd100kstyle", "bdd100k.json")
    ls_dst = join(ann_root, "cocostyle",
                  "bdd100k_{}_ls_coco.json".format(split))
    ds_dst = join(ann_root, "cocostyle",
                  "bdd100k_{}_ds_coco.json".format(split))
    all_dst = join(ann_root, "cocostyle",
                   "bdd100k_{}_all_coco.json".format(split))

    json_paths = [ls_dst, ds_dst]

    tables_dir = join(home, "inequity-release", "tables")
    shutil.rmtree(tables_dir, ignore_errors=True)
    os.makedirs(tables_dir)

    # Generate Table 2
    # Dataset: BDD100K All Person
    # Model: Faster R-CNN
    # Backbone: R-50-FPN
    # Weights: MS COCO/Unweighted
    if gen_flags[2]:
        bdd_to_coco(bdd_src, all_dst, img_dir, None)
        model_names = ["e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml"]
        res = run_trials(home,
                         root,
                         img_dir,
                         [all_dst],
                         model_names,
                         EvalMode.COCO)
        print_stats(res, model_names, "2_{}".format("mscoco"))
        model_names = ["e2e_faster_rcnn_R_50_FPN_1x_cocostyle.yaml"]
        weight_dirs = ["weighted1"]
        res = run_trials(home,
                         root,
                         img_dir,
                         [all_dst],
                         model_names,
                         EvalMode.FINAL,
                         10,
                         weight_dirs=weight_dirs)
        print_stats(res, model_names, "2_{}".format("bdd100k"), weight_dirs)

    # Generate Table 3
    # Dataset: BDD100K Val
    # Model: Faster R-CNN
    # Backbone: R-50-FPN
    # Weights: MS COCO/Unweighted
    if gen_flags[3]:
        bdd_to_coco(bdd_src, ls_dst, img_dir, "LS")
        bdd_to_coco(bdd_src, ds_dst, img_dir, "DS")
        model_names = ["e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml"]
        res = run_trials(home,
                         root,
                         img_dir,
                         json_paths,
                         model_names,
                         EvalMode.COCO)
        print_stats(res, model_names, "3_{}".format("mscoco"))

        model_names = ["e2e_faster_rcnn_R_50_FPN_1x_cocostyle.yaml"]
        weight_dirs = ["weighted1"]
        res = run_trials(home,
                         root,
                         img_dir,
                         json_paths,
                         model_names,
                         EvalMode.FINAL,
                         10,
                         weight_dirs=weight_dirs)
        print_stats(res, model_names, "3_{}".format("bdd100k"), weight_dirs)

    # Generate Table 4
    # Dataset: BDD100K Val
    # Model: All
    # Backbone: All
    # Weights: MS COCO
    if gen_flags[4]:
        bdd_to_coco(bdd_src, ls_dst, img_dir, "LS")
        bdd_to_coco(bdd_src, ds_dst, img_dir, "DS")
        res = run_trials(home,
                         root,
                         img_dir,
                         json_paths,
                         MODELS,
                         EvalMode.COCO)
        print_stats(res, MODELS, "4")

    # Generate Table 5
    # Dataset: BDD100K Val (No occluded)
    # Model: Faster R-CNN
    # Backbone: R-50-FPN
    # Weights: MS COCO
    if gen_flags[5]:
        bdd_to_coco(bdd_src, ls_dst, img_dir, "LS", ignore_occluded=True)
        bdd_to_coco(bdd_src, ds_dst, img_dir, "DS", ignore_occluded=True)
        res = run_trials(home,
                         root,
                         img_dir,
                         json_paths,
                         MODELS,
                         EvalMode.COCO)
        print_stats(res, MODELS, "5")

    # Generate Table 6
    # Dataset: BDD100K Val
    # Model: Faster R-CNN
    # Backbone: R-50-FPN
    # Weights: weighted 1, 2, 3, 5, 10
    if gen_flags[6]:
        bdd_to_coco(bdd_src, ls_dst, img_dir, "LS")
        bdd_to_coco(bdd_src, ds_dst, img_dir, "DS")
        weight_dirs = ["weighted1",
                       "weighted2",
                       "weighted3",
                       "weighted5",
                       "weighted10"]
        model_names = ["e2e_faster_rcnn_R_50_FPN_1x_cocostyle.yaml"]
        res = run_trials(home,
                         root,
                         img_dir,
                         json_paths,
                         model_names,
                         EvalMode.FINAL,
                         10,
                         weight_dirs=weight_dirs)
        print_stats(res, model_names, "6", weight_dirs)

    # Generate Table 7
    # Dataset: BDD100K Val
    # Model: Faster R-CNN
    # Backbone: R-50-FPN
    # Weights: Unweighted
    # Daytime vs. Night
    if gen_flags[7]:
        model_names = ["e2e_faster_rcnn_R_50_FPN_1x_cocostyle.yaml"]

        time_of_day = "daytime"
        bdd_to_coco(bdd_src,
                    ls_dst,
                    img_dir,
                    "LS",
                    time_of_day=time_of_day)
        bdd_to_coco(bdd_src,
                    ds_dst,
                    img_dir,
                    "DS",
                    time_of_day=time_of_day)
        weight_dirs = ["weighted1"]
        res = run_trials(home,
                         root,
                         img_dir,
                         json_paths,
                         model_names,
                         EvalMode.FINAL,
                         10,
                         weight_dirs=weight_dirs)
        print_stats(res, model_names, "7_{}".format(time_of_day), weight_dirs)

        time_of_day = "night"
        bdd_to_coco(bdd_src,
                    ls_dst,
                    img_dir,
                    "LS",
                    time_of_day=time_of_day)
        bdd_to_coco(bdd_src,
                    ds_dst,
                    img_dir,
                    "DS",
                    time_of_day=time_of_day)
        res = run_trials(home,
                         root,
                         img_dir,
                         json_paths,
                         model_names,
                         EvalMode.FINAL,
                         10,
                         weight_dirs=weight_dirs)
        print_stats(res, model_names, "7_{}".format(time_of_day), weight_dirs)

    # Generate Figure 4
    # Dataset: BDD100K Val
    # Model: Faster R-CNN
    # Backbone: R-50-FPN
    # Weights: Unweighted
    if gen_flags[8]:
        bdd_to_coco(bdd_src, ls_dst, img_dir, "LS")
        bdd_to_coco(bdd_src, ds_dst, img_dir, "DS")
        weight_dirs = ["weighted1"]
        model_names = ["e2e_faster_rcnn_R_50_FPN_1x_cocostyle.yaml"]
        res = run_trials(home,
                         root,
                         img_dir,
                         json_paths,
                         model_names,
                         EvalMode.ALL,
                         10,
                         weight_dirs=weight_dirs)
        generate_graphs(home, res)

    # Generate images
    # Dataset: BDD100K Val
    # Model: Faster R-CNN
    # Backbone: R-50-FPN
    # Weights: Unweighted
    if gen_flags[9]:
        bdd_to_coco(bdd_src, ls_dst, img_dir, "LS")
        bdd_to_coco(bdd_src, ds_dst, img_dir, "DS")
        weight_dirs = ["weighted1"]
        model_names = ["e2e_faster_rcnn_R_50_FPN_1x_cocostyle.yaml"]

        res = run_trials(home,
                         root,
                         img_dir,
                         json_paths,
                         model_names,
                         EvalMode.FINAL,
                         10,
                         weight_dirs=weight_dirs)


def main():
    gen_flags = {
        2: True,
        3: True,
        4: True,
        5: True,
        6: True,
        7: True,
        8: True,
        9: True
    }

    home = expanduser("~")
    split = "val"
    subsets = ["ls", "ds"]
    run(home, gen_flags, split, subsets)


if __name__ == "__main__":
    main()
