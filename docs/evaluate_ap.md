训练

```
python -m torch.distributed.launch --nproc_per_node=4 train_dist.py --basenet-checkpoint weights/hourglass_104_renamed.pth --warmup --checkpoint-whole link2checkpoints_storage/PoseNet_158_epoch.pth --resume --no-pretrain --sqrt-re --weight-decay 0 --min-scale 0.3 --max-scale 1.8
```

测试

```
python -u /home/jia/Desktop/OffsetGuided/evaluate.py --no-pretrain --initialize-whole False --checkpoint-whole link2checkpoints_storage/PoseNet_178_epoch.pth --resume --sqrt-re --long-edge 640 --batch-size 16 --loader-workers 8 --n-images-val 2000
```

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.566
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.795
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.605
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.546
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.601
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.606
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.816
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.640
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.566
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.666
```

测试边长换成512

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.539
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.775
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.568
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.478
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.629
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.581
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.792
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.609
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.499
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.696
```

若'--dist-max  15'

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.595
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.814
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.640
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.554
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.657
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.636
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.834
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.675
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.574
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.724
```

若'--dist-max  20'

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.600
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.815
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.647
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.550
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.675
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.643
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.837
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.682
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.572
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.743
```

若'--dist-max  25'

```
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.599
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.815
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.646
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.545
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.680
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.644
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.838
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.683
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.569
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.750
```

132 epoch, l2 loss for hmp, '--dist-max  20', 640 length

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.595
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.815
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.639
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.547
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.664
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.638
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.835
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.678
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.570
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.734
```

