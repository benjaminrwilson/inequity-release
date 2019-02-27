# Predictive Inequity in Object Detection

### Paper

http://jamiemorgenstern.com/papers/pid.pdf

[Benjamin Wilson](https://github.com/benjaminrwilson), [Judy Hoffman](https://people.eecs.berkeley.edu/~jhoffman/), [Jamie Morgenstern](http://jamiemorgenstern.com)

## Prerequisites
- Docker
- NVIDIA GPU (we used an NVIDIA V100)
- NVIDIA Drivers
- NVIDIA Docker

## Getting Started
### Installation

- Install Docker: https://docs.docker.com/install/
- Install NVIDIA Docker: https://github.com/NVIDIA/nvidia-docker

- Pull the docker image:

```
docker pull benjaminrwilson/inequity-release:latest
```

- Run the docker image:

```
bash docker/run.sh
```

Once you're within the container, you will need to get the necessary data to run the experiments listed in the paper. You will need to get the annotations (provided by us in MS COCO format), the images from the BDD100K dataset, and lastly the weights we used.

- Download the annotations, images, and weights:

```
bash scripts/get_data.sh
```

### Evaluation

- Run the evaluation (this will likely take over an hour).
```
python eval.py
```

The tables from the paper will be output as text based tables in a new folder called ```tables```. The graph will be created in a folder called ```figs```.

### Training

- If you would like to train Faster R-CNN from ImageNet initialization, we have provided a training script to train at different weights. First, make a directory as such:

```
mkdir ~/weights/
```

- Link the ```datasets``` directory as such:

```
ln -s datasets ~/github/maskrcnn-benchmark/
```

- Edit the args in ```inequity/scripts/train_large.sh``` as needed. ```augmented_loss_weights``` is a list which corresponds to the weighting put on ```["LS", "DS", "Not a Person", "A person, cannot determine skin type"]```. For example, ```[1, 5, 1, 1]``` would put weight ```5``` on individuals labeled as ```DS``` in the classification network loss of Faster R-CNN (as described in the appendix of the paper).

- To run training:

```
bash inequity/scripts/train_large.sh
```
