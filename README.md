## Prerequisites

- This project is implemented in Pytorch (>1.8). Thus, please install PyTorch first.
- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode.
- For those who failed to install ctcdecode (and it always does), you can download [ctcdecode here](https://drive.google.com/file/d/1LjbJz60GzT4qK6WW59SIB1Zi6Sy84wOS/view?usp=sharing), unzip it, and try `cd ctcdecode` and `pip install .`
- Pealse follow [this link](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) to install pytorch geometric
- You can install other required modules by conducting
  `pip install -r requirements.txt`

## Data Preparation

1. PHOENIX14 dataset: Download the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX/).
2. PHOENIX14-T datasetDownload the RWTH-PHOENIX-Weather 2014 Dataset [[download link]](https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/)
3. CSL-Daily dataset： Request the CSL Dataset from this website [[download link]](https://ustc-slr.github.io/openresources/cslr-dataset-2015/index.html)

Download datasets and extract them.
You don't need any further data preprocessing.


### Weights

Here we provide the best performance checkpoints.

| Dataset    | Backbone | Dev WER | Test WER | Pretrained model                                                                                                          |
| ---------- | -------- | ------- | -------- | ------------------------------------------------------------------------------------------------------------------------- |
| Phoenix14  | Resnet34 | 16.98   | 17.51    | [[Google Drive]](https://drive.google.com/drive/folders/1GIRjSSunMGwgOp8JqlK3x7ERct4nv5tf?dmr=1&ec=wgc-drive-globalnav-goto) |
| Phoenix14-T | Resnet34 | 16.68   | 18.23    | [[Google Drive]](https://drive.google.com/drive/folders/102_9th1pHyiv698qx6lznP7PMX07nIY3?dmr=1&ec=wgc-drive-globalnav-goto) |
| CSL-Daily  | Resnet34 | 25.08  | 24.27    | [[Google Drive]](https://drive.google.com/drive/folders/1xTAaS70KTHLajONwEJSL4npcWXguulwp?dmr=1&ec=wgc-drive-globalnav-goto) |


### Training

To train the DMLG model, please choose the dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml first, and run the command below：

`python main.py --device your_device`


### Evaluate

​To evaluate the DMLG model, choose the dataset from phoenix2014/phoenix2014-T/CSL/CSL-Daily in line 3 in ./config/baseline.yaml first, and run the command below：   
`python main.py --device your_device --load-weights path_to_weight.pt --phase test`



### Thanks

This repo is based on  [SignGraph(CVPR 2024)](https://github.com/gswycf/SignGraph).
