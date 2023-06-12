# ML-MWN
Official code for the Multi-Label Meta Weighting for Long-Tailed Dynamic Scene Graph Generation (ICMR 2023) by Shuo Chen, Yingjun Du, Pascal Mettes, and Cees Snoek.

If you use this code for a paper, please cite:


```
@inproceedings{Chen2023Multi,
  author       = {Shuo Chen and
                  Ying{-}Jun Du and
                  Pascal Mettes and
                  Cees G. M. Snoek},
  title        = {Multi-Label Meta Weighting for Long-Tailed Dynamic Scene Graph Generation},
  booktitle    = {{ICMR}},
  year         = {2023},
}
```


## Requirements
PyTorch 1.1.0+


## Training
```bash
python meta_train.py -mode predcls -data_path <action_genome_path> -nepoch 20 -bce_loss -optimizer sgd -lr 0.001
```

## Evaluation
```bash
python meta_test.py -mode predcls -data_path /<action_genome_path> -model_path <trained_model_path>
```

## Acknowledgments
This repo is based on [STTran](https://github.com/yrcong/STTran), [Meta-Weight-Net_Code-Optimization](https://github.com/ShiYunyi/Meta-Weight-Net_Code-Optimization) and [meta-weight-net](https://github.com/xjtushujun/meta-weight-net). We thank the authors for their work.
