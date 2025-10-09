# Datathon-TM-210-
Group TM-210

Name: Duong Kien Khai, Stephen Manickam S/O Alakesan, CHONG JIN WEI

This repository includes:

1. Academic Report (Academic_Report.docx)

2. data_exploration_and_analysis.py file (show all analyses and patterns found)

3. baseline_classifier.py containing our experimentations with various different models to select the best ones

4. xgb_opt.json (main model) and xgb_base.json containing model weights for inference

5. data_training.py for code that was used to train the model

6. inference.py (main inference file)

7. inference_backup.py (back-up inference file)

To run the inference, use this command in the command line:

```bash
python inference.py data.csv
```

where ```data.csv``` could be any .csv file with the following format:
- The first row contains the name of the 24 input features, optimally in the given order: {b, e, AC, FM, UC, DL, DS, DP, DR, LB, ASTV, MSTV, ALTV, MLTV, Width, Min, Max, Nmax, Nzeros, Mode, Mean, Median, Variance, Tendency}
- The second row onwards contains the corresponding data, each row represents an instance to be predicted.
- There should be 24 columns only. The target column should be excluded.

After running the inference, the predictions would be printed to ```predictions.txt```, each line contains a number corresponding to the prediction of an instance (in the order given in the .csv file), where:
- 0 = Normal
- 1 = Suspect
- 2 = Pathologic

### Case-by-case Explanation

By default, after running the command above, the program will draw a SHAP plot explaining the first sample (first row) in the data.csv file: how each feature contributes to the first sample's predicted class.

The command line can be used to further customize this behaviour:

```bash
python inference.py data.csv [class_idx=-1] [sample_idx=0]
```
where:
* ```class_idx```: the class for which the SHAP plot will show (```0```, ```1```, or ```2```). When ```class_idx=-1``` (by default), the SHAP plot shows how each feature contributes to the sample's **predicted class**.
* ```sample_idx```: the index of the sample to draw the SHAP plot for. By default, it is ```0```, representing the first sample in data.csv.

For example,
```bash
python inference.py data.csv 2
```

will draw the SHAP plot for the PATHOLOGIC class in the model's prediction for the first sample, and


```bash
python inference.py data.csv -1 1
```

will draw the SHAP plot for the predicted class of the second sample.