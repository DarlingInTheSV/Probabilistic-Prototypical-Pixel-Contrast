# Reducing Semantic Ambiguity In Domain Adaptive Semantic Segmentation Via Probabilistic Prototypical Pixel Contrast
This is the official implementation of the paper "Reducing Semantic Ambiguity In Domain Adaptive Semantic Segmentation Via Probabilistic Prototypical Pixel Contrast"

![Visualization](gifs/g2c.gif)
![Visualization](gifs/c2d.gif)
## Installation
```bash
# create conda environment
conda create --name PPPC -y python=3.8
conda activate PPPC
conda install -y ipython pip

# Upgrade pip, otherwise the installation of mmcv-full will be slow. 
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Preparation
### Download Datasets
- GTAV: Download GTAV from [here](https://download.visinf.tu-darmstadt.de/data/from_games/) and extract them to `data/gta`.
- Synthia: Download Synthia from [here](http://synthia-dataset.net/downloads/) and extract it to `data/synthia`.
- Cityscapes: Download Cityscapes from [here](https://www.cityscapes-dataset.com/downloads/) and extract it to `data/synthia`.
- Dark Zurich: Download Dark Zurich from [here](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/) and extract it to `data/synthia`.

The folder structure should like this:
```bash
PPPC
├── ...
├── data
│   ├── cityscapes
│   │   ├── leftImg8bit
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── gtFine
│   │   │   ├── train
│   │   │   ├── val
│   ├── dark_zurich
│   │   ├── gt
│   │   │   ├── val
│   │   ├── rgb_anon
│   │   │   ├── train
│   │   │   ├── val
│   ├── gta
│   │   ├── images
│   │   ├── labels
│   ├── synthia 
│   │   ├── RGB
│   │   ├── GT
│   │   │   ├── LABELS
├── ...
```
Perform preprocessing to convert label IDs to the train IDs and gather dataset statistics:

```bash
python tools/convert_datasets/gta.py data/gta --nproc 20
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 20
```
## Evaluate
### Evaluation on Cityscapes
To evaluate the model on Cityscapes, run:
```bash
python -m tools.test /path/to/config /path/to/checkpoint --eval mIoU
```
Our trained model and config are available via [GTAV &rarr; Cityscapes](https://drive.google.com/drive/folders/19Wa5oLEzO2h4I3_crvv-fzw9UMVRA48V?usp=sharing), [SYNTHIA &rarr; Cityscapes](https://drive.google.com/drive/folders/1GajYwzrOY4VLA7AVVRqLMg6p32nBGFkW?usp=sharing).

### Evaluation on Dark Zurich
- Please follow the instructions in [SePiCo](https://github.com/BIT-DA/SePiCo), and submit them to the official [test server](https://codalab.lisn.upsaclay.fr/competitions/3783). 
- Our trained model and config are available via [Cityscapes &rarr; Dark Zurich](https://drive.google.com/drive/folders/1iZ9BgflJRMeR_ejJv4zEmprdBYFExZp6?usp=sharing).
- Our submission and score log are available [here](https://drive.google.com/drive/folders/1__adce5xYsHuHZR5kQX9lgA03zdKDtiw?usp=sharing).

## Train


The detail of train configration is at 'experiments.py'.
```bash
python run_experiments.py --exp <exp_id>
```
| `<exp_id>` | task    |
|:----------:|:--------| 
|    `1`     | GTAV &rarr; Cityscapes  |
|    `2`     | Cityscapes &rarr; SYNTHIA  | 
|    `3`     | Cityscapes &rarr; Dark Zurich | 

## Acknowledgments
This project is based on the following open-source projects. We thank their authors for making the source code publicly available.
- [SePiCo](https://github.com/BIT-DA/SePiCo)
- [DaFormer](https://github.com/lhoyer/DAFormer)
- [PRCL](https://github.com/Haoyu-Xie/PRCL)