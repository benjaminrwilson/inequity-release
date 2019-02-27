import json
import os
import shutil
import sys
from enum import Enum
from os import listdir
from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tabulate import tabulate

from demo.predictor import COCODemo
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_nms

CFG_SKELETON = cfg.clone()

COCO_HEADERS = [
    "AP (%)",
    "AP50 (%)",
    "AP75 (%)"
]

FITZ_TO_INDEX = {
    "LS": 0,
    "DS": 1,
    "Not a person": 2,
    "A person, cannot determine skin type": 3
}

INDEX_TO_FITZ = {v: k for k, v in FITZ_TO_INDEX.items()}


MODELS = [
    "e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
    "e2e_faster_rcnn_R_50_FPN_1x_caffe2.yaml",
    "e2e_faster_rcnn_R_101_FPN_1x_caffe2.yaml",
    "e2e_faster_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml",
    "e2e_mask_rcnn_R_50_C4_1x_caffe2.yaml",
    "e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml",
    "e2e_mask_rcnn_R_101_FPN_1x_caffe2.yaml",
    "e2e_mask_rcnn_X_101_32x8d_FPN_1x_caffe2.yaml"
]


class EvalMode(Enum):
    COCO = 0
    FINAL = 1
    ALL = 2


def get_model(home, model_name, threshold=0.0, weights_path=None, mode=0):
    """Get the PyTorch model with the given
    model_name and load in the weights.

    Parameters
    ----------
    home : str
        Home directory in terminal.
    model_name : str
        Model name used for evaluation.
    threshold : float
        Mode for evaluation.
    weights_path : str
        The path to the weights file for evaluation.
    mode : EvalMode (Enum)
        Mode for evaluation.

    Returns
    -------
    COCODemo

    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root = join(home, "github/maskrcnn-benchmark/configs/bdd100k")

    if mode == EvalMode.COCO:
        root = root.replace("bdd100k", "")
        root += "caffe2"
    config_file = join(root, model_name)
    args = ["MODEL.DEVICE", device]
    if weights_path is not None:
        args += ["MODEL.WEIGHT", weights_path]
    cfg = CFG_SKELETON.clone()
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args)
    coco_demo = COCODemo(
        cfg,
        min_image_size=800,
        confidence_threshold=threshold,
    )
    return coco_demo


def coco2bdd(labels):
    """Map PyTorch Tensor of MS COCO
    labels to corresponding BDD100K class.

    Parameters
    ----------
    labels : Tensor
        A list of labels for predicted bboxes.

    Returns
    -------
    Tensor

    """
    person_mask = labels == 1
    labels[person_mask] = 5
    return labels


def prepare_detections(img_id, detections, mode):
    """Prepare the detections for MS COCO evaluation.

    Parameters
    ----------
        img_id : int
            The correspond img_id for a set of detections.
        detections : Tensor
            The detections for a given image.
        mode : EvalMode (Enum)
            Mode for evaluation.

    Returns
    -------
    Tensor

    """
    bboxes = detections.convert("xywh").bbox
    scores = detections.get_field("scores").unsqueeze(1)
    labels = detections.get_field("labels").float().unsqueeze(1)
    labels = coco2bdd(labels) if mode == EvalMode.COCO else labels

    id_column = torch.zeros([scores.shape[0], 1]).fill_(img_id)
    detections = torch.cat(
        (id_column, bboxes, scores, labels), dim=1).cpu().numpy().tolist()
    return detections


def print_stats(results,
                model_names,
                table_num,
                weight_dirs=["MS COCO"],
                num_stats=3):
    """Print the stats for the given experiment and
    writes them to the tables directory.

    Parameters
    ----------
        img_id : int
            The correspond img_id for a set of detections.
        model_names : list of str
            List of model names used for evaluation.
        table_num : int
            Integer mapping to a unique table.
        weight_dirs : list of str
            The names of weight directories used in the experiment.
        num_stats : int
            Integer representing how many of default MS COCO stats
            to return (ex. 3 return AP, AP50, AP75).

    Returns
    -------
    None

    """
    with open("../tables/table_{}.txt".format(table_num), "w") as table_file:
        if num_stats < 1 or num_stats > 12:
            print("Invalid num_stats size!")
            exit(0)

        num_weightings, num_models = results.shape[:2]
        means = np.round(np.mean(results, axis=2) * 100,
                         1)[..., :num_stats].reshape(-1, 2, num_stats)
        means = np.dstack((means[:, 0, :], means[:, 1, :])
                          ).reshape(num_weightings, num_models, num_stats * 2)

        stds = np.round(np.std(results, axis=2) * 100,
                        1)[..., :num_stats].reshape(-1, 2, num_stats)
        stds = np.dstack((stds[:, 0, :], stds[:, 1, :])
                         ).reshape(num_weightings, num_models, num_stats * 2)

        table = "Means:\n" + \
            generate_table(means, stds, model_names, weight_dirs) + "\n"
        print(table)
        table_file.write(table)


def get_weight_paths(root, weight_dir, trial_num, mode):
    """Get the weight paths for a given mode.

    Parameters
    ----------
        root : str
            The root directory for the weight directories.
        weight_dir : str
            The name of a given weight directory.
        trial_num : int
            The number corresponding to which trial
            is being evaluated.
        mode : EvalMode (Enum)
            Mode for evaluation.

    Returns
    -------
    list of str

    """
    if mode == EvalMode.COCO:
        return [None]
    elif mode == EvalMode.FINAL:
        return [join(root, weight_dir, str(trial_num), "model_final.pth")]
    weight_paths = []
    trial_dir = join(root, weight_dir, str(trial_num))
    for file_name in listdir(trial_dir):
        if file_name.endswith(".pth"):
            weight_paths.append(join(trial_dir, file_name))
    return sorted(weight_paths)


def get_headers(use_latex):
    """Get the headers for the results table.

    Returns
    -------
    list of str

    """
    if use_latex:
        res = ["Model", "Backbone", "Weights"]
    else:
        res = ["Model", "Backbone"]
    for header in COCO_HEADERS:
        header_ls = header + " LS"
        header_ds = header + " DS"
        res += [header_ls, header_ds]
    return res


def generate_table(means,
                   stds,
                   model_names,
                   weight_dirs,
                   use_latex=False):
    """Format and generate full table of results.

    Parameters
    ----------
        means : ndarray
            Means from experiments.
        stds : ndarray
            Standard deviations from experiments.
        model_names : list of str
            Model names for each row.
        weight_dirs : list of str
            Weight directories for each row.
        use_latex : bool
            Output table in LaTeX

    Returns
    -------
    str

    """
    table_fmt = "grid"
    if use_latex:
        table_fmt = "latex_raw"

    headers = get_headers(use_latex)

    formatted_stats = format_table(
        means, stds, model_names, weight_dirs, use_latex)
    table = tabulate(formatted_stats,
                     headers,
                     tablefmt=table_fmt,
                     floatfmt=".1f")
    if use_latex:
        table = insert_bold(means, stds, table, use_latex)

        lines = table.splitlines()
        table = ""
        threshold = 2 if use_latex else 3
        for line in lines:
            count = 0
            res = ""
            for j, char in enumerate(line):
                if char == "&":
                    if use_latex:
                        is_valid = count % 2 == 1 and count < 7
                    else:
                        is_valid = count % 2 == 0
                    if count > threshold and is_valid:
                        char = "&&"
                    if count == 7:
                        char = ""
                    count += 1
                res += char
            table += res + "\n"
    return table


def insert_bold(means, stds, table, use_latex):
    """Draw predictions and ground truth annotations
    on the image.

    Parameters
    ----------
        means : ndarray
            Means from experiments.
        stds : ndarray
            Standard deviations from experiments.
        table : str
            Tabulate table previously generated.
        use_latex : bool
            Output table in LaTeX

    Returns
    -------
    str

    """
    means = means.reshape(-1, means.shape[-1])
    stds = stds.reshape(-1, stds.shape[-1])

    lines = table.splitlines()
    for i, row in enumerate(means):
        for j in range(0, len(row) - 1, 2):
            if row[j] > row[j + 1]:
                max_mean = str(row[j])
                max_std = str(stds[i][j])
            else:
                max_mean = str(row[j + 1])
                max_std = str(stds[i][j + 1])

            if max_std == "0.0":
                plus_minus_sym = "$\pm$" if use_latex else "±"
                max_std = " {} 0.0 ".format(plus_minus_sym)
                std_rep = " "
            else:
                std_rep = "\\textbf{" + max_std + "}"
            lines[i + 4] = lines[i +
                                 4].replace(max_mean, "\\textbf{" + max_mean + "}")
            lines[i + 4] = lines[i + 4].replace(max_std, std_rep)
    new_table = ""
    for line in lines:
        new_table += line + "\n"
    return new_table


def generate_graphs(home, res):
    """Generate and save the graphs shown in the paper.

    Parameters
    ----------
        res : ndarray
            The results from the experiments

    Returns
    -------
    None

    """
    plt.style.use('ggplot')

    data = np.mean(res, axis=2).reshape(7, 2, 12)

    iterations = np.array([500, 1000, 1500, 2000, 2500, 3000, 3350]) / 1000
    ap_values_a = data[:, 0, :3]
    ap_values_b = data[:, 1, :3]

    fig, ax = plt.subplots(1, 3, figsize=[12, 3])
    labels = ['AP', 'AP$_{50}$', 'AP$_{75}$']
    for i in range(ap_values_a.shape[1]):
        ax[i].plot(iterations, ap_values_a[:, i], linewidth=3)
        ax[i].plot(iterations, ap_values_b[:, i], linewidth=3)
        ax[i].set_ylabel(labels[i], fontsize=22)
        ax[i].set_xlabel('Iterations (K)', fontsize=22)
        ax[i].tick_params(axis='both', which='major', labelsize=18)
        if i == 0:
            ax[i].legend(['LS', 'DS'], fontsize=12)

    plt.tight_layout()
    figs_dir = join(home, "inequity-release", "figs")
    shutil.rmtree(figs_dir, ignore_errors=True)
    os.makedirs(figs_dir)
    fig.savefig(join(figs_dir, "BDDval_ap_vs_iter.pdf"))


def format_table(means, stds, model_names, weight_dirs, use_latex):
    """Format the table before full generation.

    Parameters
    ----------
        means : ndarray
            Means from experiments.
        stds : ndarray
            Standard deviations from experiments.
        model_names : list of str
            Model names for each row.
        weight_dirs : list of str
            Weight directories for each row.
        use_latex : bool
            output table in LaTeX

    Returns
    -------
    list of lists

    """
    res = []

    plus_minus_sym = "$\pm$" if use_latex else "±"
    for i, weight_dir in enumerate(weight_dirs):
        for j, model_name in enumerate(model_names):
            elems = model_name.split("_")[1:-1]
            model = elems[0].capitalize() + " "
            model += elems[1].upper().replace("CNN", "-CNN")
            backbone = ""
            for k, elem in enumerate(elems[2:-1]):
                backbone += elem
                if k < len(elems[2:-1]) - 1:
                    backbone += "-"

            stats = []
            for mean, std in zip(means[i, j], stds[i, j]):
                stats.append(
                    str(mean) + " {} ".format(plus_minus_sym) + str(std))
            if use_latex:
                model = "\\textbf{" + model + "}"
                backbone = "\\textbf{" + backbone + "}"
                row = [model, backbone] + stats
            else:
                row = [model, backbone, weight_dir] + stats
            res.append(row)

        if len(model_names) > 1:
            target_shape = means.shape[-1]
            average = np.round(means.reshape(-1, target_shape).mean(axis=0), 1)
            res.append(["Average", "-", "-"] + average.tolist())
    return res


def draw_predictions(img, img_name, predictions, coco_gt, img_id):
    """Draw predictions and ground truth annotations
    on the image.

    Parameters
    ----------
        img : ndarray
            Image as input from OpenCV (B, G, R)
        img_name : str
            Name of the image.
        predictions : BoxList
            BoxList of prediction information.
        coco_gt : COCO
            Instance of COCO object loaded with
            ground truth annotations.
        img_id : int
            The id mapping to a unique image.

    Returns
    -------
    None

    """

    predictions = boxlist_nms(predictions, .3, score_field="scores")

    person_mask = predictions.get_field("labels") == 5
    predictions = predictions[person_mask]

    score_mask = predictions.get_field("scores") > .85
    predictions = predictions[score_mask]

    ann_ids = coco_gt.getAnnIds(img_id)
    anns = coco_gt.loadAnns(ann_ids)

    gt_boxes = torch.Tensor([ann["bbox"] for ann in anns])
    gt_labels = torch.Tensor([ann["category_id"] for ann in anns])
    fitz_cats = torch.Tensor(
        [FITZ_TO_INDEX[ann["fitz_category"]] for ann in anns])

    height, width = img.shape[:2]
    targets = BoxList(gt_boxes, (height, width), mode="xywh").convert("xyxy")
    targets.add_field("labels", gt_labels)
    targets.add_field("fitz_cats", fitz_cats)

    targets = targets[gt_labels == 5]
    if targets.bbox.shape[0] == 0:
        return

    predictions, tps = get_tps(predictions, targets.bbox, img_name)

    bboxes = predictions.bbox
    labels = predictions.get_field("labels")
    scores = predictions.get_field("scores")

    box_thickness = 6
    for pred, label, score, tp in zip(bboxes, labels, scores, tps):
        x1, y1, x2, y2 = pred.split(1, dim=-1)
        label = label.item()
        score = score.item()

        if tp == 0:
            color = (0, 0, 255)
        elif tp == 1:
            color = (0, 255, 255)
        elif tp == 2:
            color = (0, 255, 0)

        img = cv2.rectangle(img,
                            (x1, y1),
                            (x2, y2),
                            color,
                            box_thickness,
                            cv2.LINE_AA)

    gt_bboxes = targets.bbox
    fitz_cats = targets.get_field("fitz_cats")
    gt_labels = targets.get_field("labels")

    font = cv2.FONT_HERSHEY_PLAIN
    for bbox, label, fitz_cat in zip(gt_bboxes, gt_labels, fitz_cats):
        x1, y1, x2, y2 = bbox.split(1, dim=-1)

        if label == 5:
            if fitz_cat == 0:
                gt_color = (203, 192, 255)
            elif fitz_cat == 1:
                gt_color = (128, 0, 128)
            else:
                gt_color = (255, 255, 255)

            img = cv2.rectangle(img,
                                (x1, y1),
                                (x2, y2),
                                gt_color,
                                box_thickness,
                                cv2.LINE_AA)

            fitz_text = INDEX_TO_FITZ[fitz_cat.item()]
            tw, th = cv2.getTextSize(fitz_text,
                                     font,
                                     2,
                                     2)[0]

            if fitz_cat == 0 or fitz_cat == 1:
                img = cv2.rectangle(img,
                                    (x2 - tw - 3, y2 - th - 3),
                                    (x2, y2),
                                    gt_color,
                                    cv2.FILLED,
                                    cv2.LINE_AA)

                img = cv2.putText(img,
                                  fitz_text,
                                  (x2 - tw, y2),
                                  font,
                                  2,
                                  (0, 0, 0),
                                  2,
                                  cv2.LINE_AA)

    dst = join("../images", img_name)
    cv2.imwrite(dst, img)


def get_tps(predictions, gt_bboxes, img_name):
    """Get the true positives (at different IoUs) from
    given a set of predictions and a set of ground truth
    annotations.

    Parameters
    ----------
        predictions : BoxList
            Set of predictions from a model.
        gt_bboxes : BoxList
            Set of ground truth annotations for an image.
        img_name : str
            The name of the image.

    Returns
    -------
    (BoxList, Tensor)

    """

    _, indices = torch.sort(predictions.get_field("scores"), descending=True)
    sorted_bboxes = predictions.bbox[indices]

    matches = torch.zeros(gt_bboxes.shape[0]) - 1
    tps = torch.zeros(sorted_bboxes.shape[0])
    for i, bbox in enumerate(sorted_bboxes):
        ious = bb_iou(bbox, gt_bboxes)
        iou, index = torch.max(ious, dim=0)
        if iou >= .5 and iou < .75:
            if matches[index] < 0:
                tps[i] = 1
                matches[index] = 1
        elif iou >= .75:
            if matches[index] < 0:
                tps[i] = 2
                matches[index] = 1
    return predictions[indices], tps


def bb_iou(bb1, bb2):
    """Given a single bounding box and a set of bounding boxes,
    this will return the ious between every pair.

    Parameters
    ----------
        bb1 : Tensor
            A single bounding box.
        bb2 : Tensor
            Multiple bounding boxes.

    Returns
    -------
    Tensor

    """
    bb1_x1, bb1_y1, bb1_x2, bb1_y2 = bb1[..., :4].split(1, dim=-1)
    bb2_x1, bb2_y1, bb2_x2, bb2_y2 = bb2[..., :4].split(1, dim=-1)

    intersection_x1 = torch.max(bb1_x1, bb2_x1)
    intersection_y1 = torch.max(bb1_y1, bb2_y1)
    intersection_x2 = torch.min(bb1_x2, bb2_x2)
    intersection_y2 = torch.min(bb1_y2, bb2_y2)

    interection_width = torch.clamp(intersection_x2 - intersection_x1 + 1, 0)
    interection_height = torch.clamp(intersection_y2 - intersection_y1 + 1, 0)
    intersection_area = interection_width * interection_height
    bb1_area = (bb1_x2 - bb1_x1 + 1) * (bb1_y2 - bb1_y1 + 1)
    bb2_area = (bb2_x2 - bb2_x1 + 1) * (bb2_y2 - bb2_y1 + 1)

    iou = intersection_area / (bb1_area + bb2_area - intersection_area)
    return iou


def disable_print():
    """Disable printing to terminal.

    Returns
    -------
    None

    """
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    """Enable printing to terminal.

    Returns
    -------
    None

    """
    sys.stdout = sys.__stdout__
