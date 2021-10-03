# HypMix for Text

### Requirements
* Python 3.6 or higher
* Pytorch >= 1.3.0
* Pytorch_transformers (also known as transformers)
* Pandas, Numpy, Pickle
* Fairseq

### Downloading and preprocessing the data

For AGNews and DBPedia, we follow [MixText](https://github.com/GT-SALT/MixText) for downloading and preprocessing the data.

The Arabic Hate Speech Classification data can be obtained from [here](https://github.com/nuhaalbadi/Arabic_hatespeech). We follow similar preprocessing as the other datasets.


### Running HypMix on AGNews in supervised setup with limited training data

```
python ./code/train.py --gpu 0 --n-labeled 10 --data-path ./data/ag_news_csv/ --batch-size 8 --batch-size-u 1 --epochs 50 --val-iteration 20 --lambda-u 0 --T 0.3 --alpha 16 --mix-layers-set 6 7 9 12 --separate-mix True --save_file_name_csv ag_news_n_10.csv
```

### Acknowledgement

Repository forked from [MixText](https://github.com/GT-SALT/MixText).