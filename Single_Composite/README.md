### Training 
 1. Download the training and testing dataset
 2. To train BioIR on a dataset, e.g., CSD, run
```
cd Single_Composite
sh train.sh options/CSD.yml
 ```

### Evaluation
1. Download the pre-trained model and place it in ```pretrain_model```
2. Download the test set, then run
```
python eval.py --data CSD
``` 
to obtain the resulting images
3. Run
```
python metrics_score.py --data CSD
``` 
to obtain the quality results

For deraining tasks, e.g., DID, use
```
python metrics_score.py --data DID --test_y_channel
```

For CDD, run
```
python cdd_metrics.py --data CDD
python cdd_metrics.py --data CDD-Base
```



#### Dataset structure

```
    |--Datasets
    |    |--CDD
    |    |    |--train
    |    |    |    |--clear
    |    |    |    |    |--00009.png
    |    |    |    |    |--00011.png
    |    |    |    |     ...
    |    |    |    |--input
    |    |    |    |    |--haze_00009.png
    |    |    |    |    |--haze_00011.png
    |    |    |    |     ...
    |    |    |--test
    |    |    |    |--clear
    |    |    |    |    |--00002.png
    |    |    |    |    |--00008.png
    |    |    |    |     ...
    |    |    |    |--input
    |    |    |    |    |--haze_00002.png
    |    |    |    |    |--haze_00008.png
    |    |    |    |     ...
    |    |--lol-blur
    |    |    |--train
    |    |    |    |--blur
    |    |    |    |    |--0000_0010.png
    |    |    |    |    |--0000_0011.png
    |    |    |    |     ...
    |    |    |    |--gt
    |    |    |    |    |--0000_0010.png
    |    |    |    |    |--0000_0011.png
    |    |    |    |     ...
    |    |    |--test
    |    |    |    |--blur
    |    |    |    |    |--0012_0010.png
    |    |    |    |    |--0012_0011.png
    |    |    |    |     ...
    |    |    |    |--gt
    |    |    |    |    |--0012_0010.png
    |    |    |    |    |--0012_0011.png
```

