# Balanced-RDD

[![license](https://camo.githubusercontent.com/4738d430387c93da0d49ef0428a7c7ddae18e81eaff99a014996d4f6b30fd3ef/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f6c6963656e73652f3a757365722f3a7265706f2e737667)](https://github.com/RichardLitt/standard-readme/blob/main/example-readmes/LICENSE) [![standard-readme compliant](https://camo.githubusercontent.com/f116695412df39ab3c98d8291befdb93af123f56aecc79fff4b20c410a5b54c7/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f726561646d652532307374796c652d7374616e646172642d627269676874677265656e2e7376673f7374796c653d666c61742d737175617265)](https://github.com/RichardLitt/standard-readme)

Balanced-RDD is a lightweight model for road damage detection that is designed to achieve the best balance of speed and accuracy.

![](/Users/heqihan/Desktop/Balanced-RDD/results.png)

## Table of Contents

- [Dataset](#Usage)
- [Install](#Install)
- [Usage](#Usage)

## Dataset

We use the Multi-Perspective Road Damage Dataset, and everything about it is in the Baidu web disk：

link: https://pan.baidu.com/s/1E-Y9tU66-1a2SncA_s7APw?pwd=1027  code: 1027

## Install

Clone repo and install [requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) in a [**Python>=3.8.0**](https://www.python.org/) environment, including [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

```
pip install -r requirements.txt  # install
```

## Usage

Training:

 ```
 from ultralytics import YOLO
 
 if __name__ == "__main__":
 
 # "From scratch"
 
   model = YOLO('ultralytics/models/Balnaced-RDD.yaml')
   
   model.train(**{'cfg':'ultralytics/yolo/cfg/default.yaml'})
 ```

![](/Users/heqihan/Desktop/Balanced-RDD/train.png)

## Statement

Ongoing update……

Email: qihanhe27@163.com
