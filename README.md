# ner-battle-ground
The necessary libraries needed for running the experiment can be found in req.txt file. The report can be found [here](https://github.com/dannyrichy/ner-battle-ground/blob/master/RISE_ner_assgn.pdf)

# Steps to run the code
## Training
To start training the two systems with "bert-base-cased" run training.py using the following command
```shell
python training.py --model_type a
```
Options for model_type are a/b denoting the two systems

Once the model is trained, there will be a folder called models created in the current working directory. Inside which there will be two folders
- output
  - sys_{a/b} : This directory will contain the results and checkpoints, the test dataset results are also stored here as json file
- sys_{a/b}
  - checkpoint-final: This directory will contain the trained model

## Inference
A separate file is provided should you decide to just load the trained model and run inference on the test dataset. Note that this will fail if there are no trained models.

```shell
python inference.py --model_type a
```
