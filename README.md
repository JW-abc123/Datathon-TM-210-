# Datathon-TM-210-
Group TM-210

Name: Duong Kien Khai, Stephen Manickam S/O Alakesan, CHONG JIN WEI

This repository includes:

1. Academic Report (Academic_Report.docx)

2. Data_exploration_and_analysis.py file (show all analysis and pattern found)

3. Baseline classifier to show all model that are trained without fine tuning 

4. xgb_opt.json (main model) and xgb_base.json for containing model weights for inference

5. data_training.py for code that are used to train the model

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

