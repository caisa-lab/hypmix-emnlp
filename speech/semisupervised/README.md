# HypMix for Semisupervised Speech Classification

Download the URDU Speech Emotion Recognition data from [here](https://github.com/siddiquelatif/URDU-Dataset) and follow [BC Learning]() to preprocess the data.

### Run semi-supervised HypMix for speech on Urdu SER
```
python main_unsup.py --data /path/to/dataset/folder/ --mixup_type sound --batchSize=32  --netType envnetv2 --BC --strongAugment
```

### Acknowledgement

Repository forked from [SpeechMix](https://github.com/midas-research/speechmix).