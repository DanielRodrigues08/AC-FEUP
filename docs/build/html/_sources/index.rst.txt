=========================
Predicting WBNA Playoffs
=========================

Welcome to the documentation for this project!


Introduction
------------

This project aims to investigate and create a model capable of predicting which teams advance to WNBA's playoffs in a given year.
It was developed in Python with the help of SKLearn library to achieve optimal results. In this project, we delved deep into all
aspects of feature engineering - using methods to select the best features for our model, creating new features and even reducing
the dimensionality of the data using PCA. We also used different models to predict the results and compared their performance, ranging
from full-fledged heterogeneous ensemble methods to simple logistic regression.

Installation
------------

To install this project, run the following commands in the root folder:

.. code-block:: bash

   pip install -r requirements.txt

Usage
-----

To run this project, follow these steps:

1. Run the notebooks script (to see the output associated to each notebook you'll need to run these files manually):

   .. code-block:: bash

      python run_notebooks.py

2. To see the graphics associated with transformation step, run `transformation.ipynb` to transform the data, inside `transformation` folder.
3. To see the graphics associated with classification step, run `classification.ipynb` to present the final results, inside `classification` folder.
