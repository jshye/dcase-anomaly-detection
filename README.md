# DCASE Anomaly Detection
> Anomalous Machine Sound Detection using DCASE challenge Task 2 Dataset.  
https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds#audio-dataset


## 1. Train
Train using only normal data.
### Command Line Arguments
* --model: Name of the model to trian. Should be implemented in ***models*** dir.
* --machine, -m: Type of machine dataset to train.
* --run_val: if set, run validation
* --save_ckpt: if set, save checkpoint at the end of every epoch
* --run_name: experiment id. Learning curve and checkpoints will be saved to './results/{run_name}'
* --ood: Type of machine to use as out-of-distribution data

    ex) Train with ToyCar data, using ToyTrain data as OOD samples.
    ```
    python train.py --model resnet -m ToyCar --ood ToyTrain --run_name test-1 --save_ckpt
    ```

## 2. Test
Run model prediction using normal & anomaly data and calculate ROC AUC.

```
python test.py --model <model-name> -m <machine-type> --run_name <result-dir-name>
```

### Test Steps
for each checkpoint saved in ***./results/<run_name>/checkpoints*** :

1. Load Model
2. Get model predictions for the source domain data.  
    2-1. Calculate anomaly score from predictions.  
    2-2. Get ROC AUC score.
3. Get model predictions for the target domain data (in domain shift adaptation tasks).
