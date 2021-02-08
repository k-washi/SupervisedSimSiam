# SupervisedSimSiam

## Task
[Food101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/)の分類


## @TODO
[ ] データローダの作成  
[ ] EfficientNetの実装  
[ ] SupervisedContrastiveLearningの実装  
[ ] SimSiamによる学習の実装  

# Setting

## データセットの準備

Food101のデータセットのダウンロードと展開

```
./script/food101_dataset.sh
```


# Ref

- [PytorchでEfficientNetを実装してみる](https://tzmi.hatenablog.com/entry/2020/02/06/183314)
- [Exploring Simple Siamese Representation Learning](https://github.com/leaderj1001/SimSiam)
- [SupContrast: Supervised Contrastive Learning](https://github.com/HobbitLong/SupContrast)

### データセット
```
@inproceedings{bossard14,
  title = {Food-101 -- Mining Discriminative Components with Random Forests},
  author = {Bossard, Lukas and Guillaumin, Matthieu and Van Gool, Luc},
  booktitle = {European Conference on Computer Vision},
  year = {2014}
}
```

# notebook

[food101_effnet_train.ipynb](https://colab.research.google.com/drive/1N_aXT_8YxamsoYlAzUEcIhtVa94KeBZ9)