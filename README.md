# ⚽ Football Wages Prediction Project 

Here's a quick breakdown of what we need to do for the Practice Assignment.

## 🎯 The Goal
We need to build a model to predict football players' wages (`log_wages`) based on their stats (skills, physical attributes, etc.). 

## 📂 The Files
*   **`midterm_assignment_template.ipynb`**: This is where the magic happens. We write our code here. Don't forget to add our names/student numbers at the top!
*   **`football_wages.csv`**: Our **training data**. Contains player stats + the `log_wages` column we are trying to predict.
*   **`football_autograder.csv`**: The **test data** for the autograder. We run our trained model on this to generate predictions.
*   **`football_practice.pdf`**: The official rules and instructions.
*   **`autograder_submission.txt`**: Where we save our final predictions to submit.


## 📝 To-Do List
1.  Open the notebook and fill in our details.
2.  Load `football_wages.csv` and clean/preprocess the data (handle that `nationality_name` b-string issue?).
3.  Train our regression models.
4.  Generate predictions for the players in `football_autograder.csv`.
5.  Save output to `autograder_submission.txt`.


## STATUS
1. The preprocessing pipeline is done. To do preprocessing, run the code below. It returns the features as a numpy array and the target column as a pandas Series.
```python
import pandas as pd
from src.preprocessing import run_pipeline_2

df = pd.read_csv('../football_wages.csv')
X_processed, y, preprocessor = run_pipeline_2(df)

```
