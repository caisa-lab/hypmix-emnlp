# HypMix for Supervised Speech Classification

Fork the [SpeechMix](https://github.com/midas-research/speechmix) repository and replace the <mark>utils.py</mark> file with the current <mark>utils.py</mark> implementing the various Möbius operations in CuPy. Follow the same repository or [BC Learning](https://github.com/mil-tokyo/bc_learning_sound) to download and preprocess the data.





### Run HypMix for speech on ESC10
```
python main.py --dataset esc10 --split <split_no> --mixup_type sound --batchSize=32  --netType envnetv2 --data ./datasets/ --BC --strongAugment
```

### Acknowledgement

Repository forked from [SpeechMix](https://github.com/midas-research/speechmix).