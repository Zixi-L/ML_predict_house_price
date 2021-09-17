# Problem description

## Background

Our goal is to build a simple ML pipeline that can preprocess data, train and evaluate a regression model to predict home prices from the attributes of homes. We expect this will take less than 3 hrs. 

## Assignment

Please design and implement three modules that can be invoked from command line. The preprocessing module should prepare the data for modeling. The dataset we will work with is a housing dataset; it’s available in the data folder. It might be helpful to explore the data in a Jupyter notebook before deciding what preprocessing steps would be useful. We want to be able to control the split between training and test sets without changing code. Think about where and how you would implement this.

The training module should train a Random Forest model and serialize it to disk. We want to be able to pass in hyperparameters without editing code. Generally, we don’t want to be hard-coding data file paths and output paths. Consider how you would manage them. 

Finally, the evaluate module should evaluate the trained model on the test set. It should log RMSE and MAE. It should also produce a simple plot of predicted vs actual home values. 

Some general pointers to consider: Think about how you would use logging to help you identify problems or confirm correctness of your code. Consider unit testing when appropriate. If you have to choose between investing your energy between good software engineering versus making things pretty (for ex plots), please choose software engineering. Correct, readable and maintainable code is more important than pretty output. You can add the libraries you need to requirements.txt before sending us your code so that we can reproduce your results. Try to time-box yourself to a maximum of 5 hours and send us your solution. 

## How to reproduce 

Install all the required packages:
pip freeze>requirement.txt

Run code:
Open terminal and run:
python3 driver.py (all the parameters have default value, but they can also be customized as: python3 driver.py --file ‘path/home_data.csv’ --outputpath ‘path’ --test_size 0.3 --parameter value)

You may also want to create a fold called result to store the output. # ML_predict_house_price
