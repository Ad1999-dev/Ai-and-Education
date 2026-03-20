# Assignment 4: Equation of a Slime

In this assignment, we'll get our hands dirty with data and create our first ML model.

## Attribution

When using or sharing this material, please include the following attribution:

> Originally developed by Hunter McNichols as part of the course COMPSCI 383 Artificial Intelligence at the University of Massachusetts Amherst.  
> Shared under the CC BY 4.0 License.

## Assignment Objectives
- Learn the basics of the Pandas and SciKit Learn Python libraries
- Learn how to analyze a dataset in a Jupyter Notebook and share your insights with others
- Experience the machine learning workflow with code
- Get first-hand exposure to performing a regression on a dataset

## Pre-Requisites
Knowledge of the basic syntax of Python is expected, as is background knowledge of the algorithms you will use in this assignment.

If part of this assignment seems unclear or has an error, please reach out via our course's CampusWire channel.

## Rubric

| Task                          | Points | Details                                                   |
|-------------------------------|--------|-----------------------------------------------------------|
| Code Runs                      | 10      | Notebook runs without error                              |
| Part 1                         | 10     | Completion of Part 1: Loading Dataset                     |
| Part 2                         | 10     | Completion of Part 2: Splitting Dataset                   |
| Part 3                         | 10     | Completion of Part 3: Linear Regression                   |
| Part 4                         | 10     | Completion of Part 4: Cross Validation                    |
| Part 5                         | 10     | Completion of Part 5: Polynomial Regression               |
| **Total Points**               | **60** |                                                           |

## Overview

It's finally happened—life on other planets! The Curiosity rover has found a sample of life on Mars and sent it back to Earth. The life takes the form of a nanoscopic blob of green slime! Scientists the world over are trying to discover the properties of this new life form.

Our team of scientists at Umass has run a number of experiments and discovered that the slime seems to react to Potassium Chloride (KCl) and heat. They've run an exhaustive series of experiments, exposing the slime to various amounts of KCl and temperatures, recording the change in size of the slime after one day.

They've gathered all the results and summarized them into this table:
[Science Data CSV](./science_data_large.csv)

Your mission is to harness the power of machine learning to determine the equation that governs the growth of this new life form. Ultimately, the discovery of this new equation could unlock some of the secrets of life and the universe itself!

## Build Your Notebook

To discover the equation of slime, we are going to take the dataset above and use the Python libraries **Pandas** and **SciKit Learn** to create a linear regression model.

Below is a sample notebook you will use as a starting point for the assignment. It includes all of the required sections and comments to explain what to do for each part. More guidance is given in the final section.

Note: When writing your output equations for your sample outputs, you can ignore values outside of 5 significant figures (e.g. 0.000003 is just 0).

## Useful Tutorials and Documentation

### Pandas

There are many different data loading/analysis libraries out there for Python, but don't reinvent the wheel. **Pandas** is by far the most universally used library for manipulating datasets. It includes tools for loading datasets, slicing/combining data, and easily transforming back and forth to NumPy primitives.

The following tutorials should cover all the tools you will need to complete this assignment. 
[How do I read and write tabular data?](https://pandas.pydata.org/docs/getting_started/intro_tutorials/02_read_write.html)
[How do I select a subset of a DataFrame?](https://pandas.pydata.org/docs/getting_started/intro_tutorials/03_subset_data.html)

The following function will also be helpful for any data mapping you need to do in the classification section.
[Pandas Replace Documentation](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html)

### SciKit Learn

**SciKit Learn** is a popular and easy-to-use machine learning library for Python. One reason why is that the documentation is very thorough and beginner-friendly. You should get familiar with the setup of the docs, as we will be using this library for multiple assignments this semester.

- Dataset splitting
[Train Test Split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
[Cross Validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)

- Regression
[Linear Regression Tutorial](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)
[Linear Model](https://scikit-learn.org/stable/modules/linear_model.html)
[Basis Functions](https://scikit-learn.org/stable/modules/linear_model.html#polynomial-regression-extending-linear-models-with-basis-functions)

## Submission

Just as with Assignment 3, please submit your GitHub Classroom information, along with an exported file of your notebook that includes outputs.