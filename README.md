## Datasets
Follow DATASET.md to install ImageNet and other 10 datasets referring to CoOp.
## Running
You can modify the configurations in the configs/[dataset].yaml, including shots, learning rate, train epoch, etc. 
Here we provide the implementation of KCL on six transfer learning models including CLIP, CoOp, CLIPAdapter, Tip-Adapter, Tip-Adapter-F and MaPLe.

You can get the performance of the model without KCL by:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/[dataset].yaml --model=[model] --shots=k
```
For example, 
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/imagenet.yaml --model=CoOp --shots=1
```

You can run KCL by:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/[dataset].yaml --model=KCL[model] --shots=k
```

For example,
```bash
CUDA_VISIBLE_DEVICES=0 python main.py --config configs/imagenet.yaml --model=KCLCoOp --shots=1
```
## Contact
Please contact yaohuili@smail.nju.edu.cn or zhouqf@smail.nju.edu.cn if you have any question about this project
