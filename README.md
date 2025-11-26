# ⚽ Football Wages Prediction Project 

Yo team! 👋

Here's a quick breakdown of what we need to do for the TI3145TU Midterm Assignment.

## 🎯 The Goal
We need to build a model to predict football players' wages (`log_wages`) based on their stats (skills, physical attributes, etc.). 

## 📂 The Files
*   **`midterm_assignment_template.ipynb`**: This is where the magic happens. We write our code here. Don't forget to add our names/student numbers at the top!
*   **`football_wages.csv`**: Our **training data**. Contains player stats + the `log_wages` column we are trying to predict.
*   **`football_autograder.csv`**: The **test data** for the autograder. We run our trained model on this to generate predictions.
*   **`football_practice.pdf`**: The official rules and instructions. Definitely read this so we don't miss points on technicalities.
*   **`autograder_submission.txt`**: Where we save our final predictions to submit.

## 🛠️ Tech Stack
Standard ML stuff. Make sure you have these installed:
*   `numpy`
*   `pandas`
*   `matplotlib`
*   `scikit-learn` (specifically `KNeighborsRegressor` and `SGDRegressor` mentioned in the template)

## 📝 To-Do List
1.  Open the notebook and fill in our details.
2.  Load `football_wages.csv` and clean/preprocess the data (handle that `nationality_name` b-string issue?).
3.  Train our regression models.
4.  Generate predictions for the players in `football_autograder.csv`.
5.  Save output to `autograder_submission.txt`.

Let's crush this! 🚀

