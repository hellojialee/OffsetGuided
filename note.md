直接在stride=4的特征图上生成label，并且立方插值回原始输入图片尺寸，然后直接找点
----------------------------------------------

```sh
python -u /home/jia/Desktop/OffsetGuided/evaluate.py --no-pretrain --initialize-whole False --checkpoint-whole link2checkpoints_storage/PoseNet_4_epoch.pth --resume --sqrt-re --batch-size 16 --loader-workers 8 --dataset val --thre-hmp 0.06 --topk 32 --headnets hmp omp --dist-max 20 --dataset val
```

Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets= 20 ] = 0.513

 Average Precision (AP) @[ IoU=0.50 | area= all | maxDets= 20 ] = 0.784

 Average Precision (AP) @[ IoU=0.75 | area= all | maxDets= 20 ] = 0.545

 Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.434

 Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.627

 Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 20 ] = 0.565

 Average Recall (AR) @[ IoU=0.50 | area= all | maxDets= 20 ] = 0.807

 Average Recall (AR) @[ IoU=0.75 | area= all | maxDets= 20 ] = 0.600

 Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.466

 Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.705

直接在stride=4的特征图上生成label，并且直接在预测的stride=4特征图上找点，最后将找到的坐标乘以stride
---------------------------------------------------------------

### 直接量化取整数点（响应峰值）

```sh
python -u /home/jia/Desktop/OffsetGuided/evaluate.py --no-pretrain --initialize-whole False --checkpoint-whole link2checkpoints_storage/PoseNet_4_epoch.pth --resume --sqrt-re --batch-size 16 --loader-workers 8 --dataset val --thre-hmp 0.06 --topk 32 --headnets hmp omp --dist-max 20 --dataset val
```

Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets= 20 ] = 0.552

 Average Precision (AP) @[ IoU=0.50 | area= all | maxDets= 20 ] = 0.802

 Average Precision (AP) @[ IoU=0.75 | area= all | maxDets= 20 ] = 0.596

 Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.473

 Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.667

 Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 20 ] = 0.603

 Average Recall (AR) @[ IoU=0.50 | area= all | maxDets= 20 ] = 0.822

 Average Recall (AR) @[ IoU=0.75 | area= all | maxDets= 20 ] = 0.642

 Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.501

 Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.747

若以上各种配置不变，**使用普通L2 loss**做训练，集训训练25个周期

```sh
python -u /home/jia/Desktop/OffsetGuided/evaluate.py --no-pretrain --initialize-whole False --checkpoint-whole link2checkpoints_storage/PoseNet_24_epoch.pth --resume --sqrt-re --batch-size 16 --loader-workers 8 --dataset val --thre-hmp 0.06 --topk 32 --headnets hmp omp --dist-max 20 --dataset val
```

Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets= 20 ] = 0.535

 Average Precision (AP) @[ IoU=0.50 | area= all | maxDets= 20 ] = 0.776

 Average Precision (AP) @[ IoU=0.75 | area= all | maxDets= 20 ] = 0.584

 Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.474

 Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.626

 Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 20 ] = 0.591

 Average Recall (AR) @[ IoU=0.50 | area= all | maxDets= 20 ] = 0.798

 Average Recall (AR) @[ IoU=0.75 | area= all | maxDets= 20 ] = 0.632

 Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.494

 Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.730





### 使用最小二乘估计精修坐标点

最小二乘估计：kernel=1, sigma=3

Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets= 20 ] = 0.401

 Average Precision (AP) @[ IoU=0.50 | area= all | maxDets= 20 ] = 0.726

 Average Precision (AP) @[ IoU=0.75 | area= all | maxDets= 20 ] = 0.379

 Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.282

 Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.577

 Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 20 ] = 0.475

 Average Recall (AR) @[ IoU=0.50 | area= all | maxDets= 20 ] = 0.756

 Average Recall (AR) @[ IoU=0.75 | area= all | maxDets= 20 ] = 0.476

 Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.322

 Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.686

最小二乘估计：kernel=2， sigma=3

Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets= 20 ] = 0.403

 Average Precision (AP) @[ IoU=0.50 | area= all | maxDets= 20 ] = 0.733

 Average Precision (AP) @[ IoU=0.75 | area= all | maxDets= 20 ] = 0.380

 Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.284

 Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.578

 Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 20 ] = 0.477

 Average Recall (AR) @[ IoU=0.50 | area= all | maxDets= 20 ] = 0.761

 Average Recall (AR) @[ IoU=0.75 | area= all | maxDets= 20 ] = 0.479

 Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.325

 Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.687

同时估计x0, y0, sigma, 设置k=1, 使用L2 loss训练25 epochs

Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets= 20 ] = 0.452

 Average Precision (AP) @[ IoU=0.50 | area= all | maxDets= 20 ] = 0.751

 Average Precision (AP) @[ IoU=0.75 | area= all | maxDets= 20 ] = 0.463

 Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.369

 Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.581

 Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets= 20 ] = 0.525

 Average Recall (AR) @[ IoU=0.50 | area= all | maxDets= 20 ] = 0.774

 Average Recall (AR) @[ IoU=0.75 | area= all | maxDets= 20 ] = 0.548

 Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.392

 Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.711





