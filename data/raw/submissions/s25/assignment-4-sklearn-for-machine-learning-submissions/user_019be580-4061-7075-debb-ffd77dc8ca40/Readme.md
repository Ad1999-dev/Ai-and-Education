[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Epwalyod)
# Assignment 4: SKLearn for Machine Learning

In this assignment, we'll get our hands dirty with data and create our first ML models.

## Assignment Objectives
- Learn the basics of the Pandas and SciKit Learn Python libraries
- Experience the full machine learning workflow in a Jupyter Notebook
- Get first-hand exposure in performing a regression on a dataset
- Demonstrate understanding of the output of a simple ML lifecycle and workflow
- Explore multiple classification models on a real dataset
- Analyze and report on the different performance of different models via a scientific report

## Pre-Requisites
Knowledge of the basic syntax of Python is expected, as is background knowledge of the algorithms you will use in this assignment.

If part of this assignment seems unclear or has an error, please reach out via our course's CampusWire channel.

<!-- ## Rubric

| Task                          | Points | Details                                                   |
|-------------------------------|--------|-----------------------------------------------------------|
| Code Runs                      | 10      | Notebook runs without error                              |
| Part 1                         | 10     | Completion of Part 1: Loading Dataset                     |
| Part 2                         | 10     | Completion of Part 2: Splitting Dataset                   |
| Part 3                         | 10     | Completion of Part 3: Linear Regression                   |
| Part 4                         | 10     | Completion of Part 4: Cross Validation                    |
| Part 5                         | 10     | Completion of Part 5: Polynomial Regression               |
| **Total Points**               | **60** |                                                           | -->

# Part 1: Equation of a Slime

## Overview

It's finally happened—life on other planets! The Curiosity rover has found a sample of life on Mars and sent it back to Earth. The life takes the form of a nanoscopic blob of green slime. Scientists the world over are trying to discover the properties of this new life form.

Our team of scientists at UMass has run a number of experiments and discovered that the slime seems to react to Potassium Chloride (KCl) and heat. They've run an exhaustive series of experiments, exposing the slime to various amounts of KCl and temperatures, recording the change in size of the slime after one day.

They've gathered all the results and summarized them into this table:
[Science Data CSV](./science_data_large.csv)

Your mission is to harness the power of machine learning to determine the equation that governs the growth of this new life form. Ultimately, the discovery of this new equation could unlock some of the secrets of life and the universe itself!

## Build Your Notebook

To discover the equation of slime, we are going to take the dataset above and use the Python libraries **Pandas** and **SciKit Learn** to create a linear regression model.

A sample notebook is provide which will serve as a starting point for the assignment. It includes all of the required sections and comments to explain what to do for each part. More guidance is given in the final section.

Note: When writing your output equations for your sample outputs, you can ignore values outside of 5 significant figures (e.g. 0.000003 is just 0).

## Documentation and Resources

### SciKit Learn

**SciKit Learn** is a popular and easy-to-use machine learning library for Python. One reason why is that the documentation is very thorough and beginner-friendly. You should get familiar with the setup of the docs, as we will be using this library for multiple assignments this semester.

- Dataset splitting
[Train Test Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
[Cross Validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)

- Regression
[Linear Regression Tutorial](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)
[Linear Model](https://scikit-learn.org/stable/modules/linear_model.html)
[Basis Functions](https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions)

### Pandas
You have become acquainted with Pandas in your previous assignment but the following tutorials may prove helpful in this assignment.

The following tutorials should cover all the tools you will need to complete this assignment. 
[How do I read and write tabular data?](https://pandas.pydata.org/docs/getting_started/intro_tutorials/02_read_write.html)
[How do I select a subset of a DataFrame?](https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html)

The following function may also be helpful for any data mapping you need to do in the classification section.
[Pandas Replace Documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html)

# Part 2: Chronic Kidney Disease Classification

## Overview
Now that you've tackled regression, let's move on to **classification** by modeling and analyzing the Chronic Kidney Disease (CKD) dataset that we cleaned in the previous assignment.

In this part of the assignment will be more open-ended. Unlike Part 1, you will explore different classification models and determine which one performs best. You will need to read through a variety of different SciKit Learn pages through the course of this assignment, but this time it's up to you to find them, or have 383GPT help you.

## Instructions
First, load the cleaned CKD dataset. For grading consistency, please use the cleaned dataset included in this assignment `ckd_feature_subset.csv` instead of your version from Assignment 3 and use `42` as your random seed. Place your code and report for this section after in the same notebook, creating code and markdown cells as needed. 

Next, you will train and evaluate the following classification models:
- Logistic Regression
- Support Vector Machines (see SVC in SKLearn)
- k-Nearest Neighbors
- Neural Networks

To measure the performance of the models, perform 5 fold cross validation using the entire dataset. Report these measurements in a table where you report the average and standard deviations. Summarize these results afterwards. Which model performed the best and why do you think that is?

Finally, experiment with a handful of different configurations for the neural network (report a minimum of 3 different settings) and report on the results in a table as you did above. Summarize your findings, which parameters made the biggest difference in the for classification?

> **💡 Tip:**  LLMs are great for transforming messy outputs into clean tables quickly

## Submission 

- To make a submission for the project, submit a pdf of sklearn_sample_notebook jupyter notebook under *Assignment 4 - SKLearn for Machine Learning* on Gradescope. 
- How to generate a pdf of your jupyter notebook:
    - On your Github repository after finishing the assignment, click on sklearn_sample_notebook.ipynb to open the markdown preview.
    - Ensure that the notebook has outputs for all the cells included
    - Use your browser's "Print to PDF" feature to save your PDF.
    - On Gradescope, please assign the pages of your pdf to the specific questions/sections outlined.