# Trajectory Predictin Using a Multi-Stream Encoder Decoder Network
<img width="1247" alt="fig1 2" src="https://user-images.githubusercontent.com/124192573/216125928-53b3b0e2-8f77-4750-8265-aac44f2c3abe.png">
![image](https://user-images.githubusercontent.com/124192573/216125532-9a8dad34-bc46-46a9-bb8d-fb8ba152f94a.png)

=======

## Requirements
We recommend using python3 and a virtual env. When you're done working on the project, deactivate the virtual environment with `deactivate`.
```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

## Task
Given a time series of FIR images from an on-board camera and detected bounding boxes of objects, predict the future locations of those objects.

## Download the dataset

Here is the structure of the dataset:
```
dataset/
    Images/
        seq_0_0001.jpg
        
        ...
    seq/
        seq_00/
            motions/
                seq_00.jpg

                ...

            KeyFrameTrajectory.txt
            seq_00.csv

        ...

    dataset_params.json
            
```
*  The `Images` folder contains all the raw FIR images
* The `seq` folder contains the following informations:
    * `motions` that includes the dense optical flow images
    * `KeyFrameTrajectory.txt` that includes the visual odemetry data
    * `seq_{SEQ_NUMBER}.csv` that has information about the bounding boxes in the current sequence
* `dataset_params.json` in this file you can specify the sequences for train, validation, and test jobs

## Quick Training

1. **Setup the experiment config**:
 You can setup your own training job by creating a new `run_{EXP_NUMBER}`. Then, you can modify the parameters inside the `net_params.json`. The network model definition as well as all the training steps are defined in the `model` folder. However, you should be able to change the network configuration just by changing the parameters in the `net_params.json`. The file looks like

```json
{
    "prediction_length": 30,
    "roi_size": 10,

    "exp_num": 5,
    "cuda": true,
    "learning_rate": 1e-3,
    "batch_size": 4,
    "save_summary_steps": 100,
    ...
    
}
```

2. **Run a Train Job**:
Once the `net_params.json` is set. You can start a training task by calling the `train.py` script from the main directory. Simply run
```
python train.py --experiment_dir ${EXP_JSON_DIR} --data_dir ${DATASET_DIR} --log_dir ${LOG_SAVE_DIR} 
```
It will instantiate a model and train it on the training set following the parameters specified in `net_params.json`. It will also evaluate some metrics on the development set.

3. **Evaluate the Training**:
After the training job is over, you check the evaluation metrices in Tensorboard.
 ```
tensorboard --logdir ./{LOG_SAVE_DIR}/{EXP_NUMBER}/{TRAINING_DATE}
```
4. **Evaluation on the Test Set**:
Once you've selected the best model and hyperparameters based on the performance on the development set, you can finally evaluate the performance of the model on the test set. Run

```
python test.py --experiment_dir {EXP_JSON_DIR} --data_dir ${DATASET_DIR} --model_dir ${LOG_SAVE_DIR} 
```
The model weights are saved in the `${LOG_SAVE_DIR}` as the `best_acc_model`
<img width="983" alt="fig3_2" src="https://user-images.githubusercontent.com/124192573/216126026-b2186a39-7fc4-4e3a-9399-55c839e6f0c1.png">

For more info and questions please contact:
Alireza Rahimpour: arahimpo@ford.com or Navid Fallahinia: navid.falahinia@gmail.com 
