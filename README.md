# FocusNet 
The implementation of our Pattern Recognition 2022 paper: "FocusNet: Classifying better by focusing on confusing classes"

Paper: https://www.sciencedirect.com/science/article/abs/pii/S003132032200190X?via%3Dihub
## Note: 
- This repository mainly relies on "[ImageNet training in PyTorch](https://github.com/pytorch/examples/tree/master/imagenet)". Therefore, it is helpful for you to refer to its document.
- The first version of our architecture was named ClonalNet, and after the second revision we changed its name to FocusNet. Therefore, **the following clonalnet is just focusnet**. 
# ImageNet training in PyTorch

This implements training of popular model architectures, such as ResNet, AlexNet, and VGG on the ImageNet dataset.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Note: the `requirements.txt` in this repository is not the same as the official requirements. If something goes wrong, please use the official requirements.  
- Download the ImageNet dataset from http://www.image-net.org/
    - Then, move validation images to labeled subfolders, using [the following shell script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
## Training

To train our network, run `clonalnet_main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
python clonalnet_main.py --data /path/to/ILSVRC2012 -a resnet18 --seed 42 --gpu 0  -ebc
                                                       resnet34
                                                       mobilenet_v2
```

The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs. 

## Validation

To evaluate our network, run `clonalnet_main.py` with the desired model architecture and the path to the ImageNet dataset:

```bash
python clonalnet_main.py --data /path/to/ILSVRC2012 -a resnet18 --seed 42 --gpu 0  -ebc -e --resume clonalnet_resnet18_model_best.pth.tar
                                                       resnet34                                     clonalnet_resnet34_model_best.pth.tar
                                                       mobilenet_v2                                 clonalnet_mobilenet_v2_model_best.pth.tar

```

## Logs
The `clonal_resnet18_from_scratch.log` and the `clonal_resnet34_from_scratch.log` are the training logs of the clonalnet_resnet18 and the clonalnet_resnet34.

## Baseline
To validate the baseline results, please run:
```bash
# resnet18 / resnet34
python main.py --paradigm baseline --data /path/to/ILSVRC2012 -a resnet18 --seed 10 -e --pretrained --gpu 0
                                                                 resnet34
# mobilenet_V2
python main.py --paradigm baseline --data /path/to/ILSVRC2012 -a mobilenet_v2 --seed 10 -e --pretrained --gpu 0 --resume models/_pytorch_pretrained_checkpoints/baseline_mobilenet_v2_model_best.pth.tar

```
## Results on ILSVRC2012
|Models|Acc@1|Acc@5|Checkpoint|
|------|-----|-----|-----|
|ResNet18|69.760|89.082|[PyTorch Pre-trained](https://pytorch.org/vision/stable/models.html)|
|ClonalNet (r18)|70.422|89.562|[Baidu](https://pan.baidu.com/s/17GAra665g3Y9Uf9l_XIffg), code:1234; [Google Driver](https://drive.google.com/file/d/1VuYREp2tWDyamjzphMeb0pGMIlVTN4Se/view?usp=sharing)|
|ResNet34|73.310|91.420|[PyTorch Pre-trained](https://pytorch.org/vision/stable/models.html)|
|ClonalNet (r34)|74.366|91.884|[Baidu](https://pan.baidu.com/s/1E-MocRLYlFUxc93_E-Ndtw), code:1234; [Google Driver](https://drive.google.com/file/d/1NfnyQMP0dy3eYNuaIfs56fFj8nG4_L9f/view?usp=sharing)|
|MobileNet_v2|65.558|86.744|[Baidu](https://pan.baidu.com/s/11f5wxVbuDtKQ2WguIPDtbw), code:1234; [Google Driver](https://drive.google.com/file/d/1EecCV14dXD9yzFNfgbcTBw_CDPLZQi6i/view?usp=sharing)|
|ClonalNet (MobileNet_v2)|66.300|87.232|[Baidu](https://pan.baidu.com/s/16aAsj3-RKIoL-k4Bydt14w); [Google Driver](https://drive.google.com/file/d/1nDfBea0GSQ4Fj8cdleRhwocJw8oO2T60/view?usp=sharing)|

you can also download more checkpoints at here: [Baidu](https://pan.baidu.com/s/1BPcyHRWokKcfpGTAiuVoug), code: 1234; [Google Driver](https://drive.google.com/drive/folders/18KBAvXccSPZDAZOjVKwLqZ9ZKGqL4RMf?usp=sharing). 
 
## Reference
If you find our work is helpful to you, please cite it:
```bash
@article{zhang2022focusnet,
  title={FocusNet: Classifying better by focusing on confusing classes},
  author={Zhang, Xue and Sheng, Zehua and Shen, Hui-Liang},
  journal={Pattern Recognition},
  pages={108709},
  year={2022},
  publisher={Elsevier}
}
```