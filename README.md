# Loan_Approval_Status_Supervised_Machine_learning_Classification.
<p align="center">
     <img width="400" height="200" src="https://user-images.githubusercontent.com/119164734/209445317-df30b4ab-a6c4-4fb0-a438-a515f3ac022d.jpg">
</p>

## 1. Problem Statement

Dream Housing Finance company deals in all kinds of home loans. They have presence across all urban, semi urban and rural areas. Customer first applies for home loan and after that company validates the customer eligibility for loan.

Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have **provided a dataset to identify the customers segments that are eligible for loan amount so that they can specifically target these customers**.

## 2.Importing libraries Package.

The following libaries and tools are used in the project.

<p align="center">
     <img width="400" height="200" src="https://user-images.githubusercontent.com/119164734/208314443-2ef2756b-470c-4619-b7a4-1a48bf3c6cd3.png">
</p>


**Pandas**: Importing for panel data analysis

**Numpy**: For numerical python operations

**Matplotlib (Pyplot)**: A popular plotting library used along with pandas

**Seaborn**: A library, built on matplotlib, to create beautiful plots

**Scikit Learn**: To perform all tasks realted to Machine Learning

**Encoder label**: Converting categorical variable to Numerical variable.

## 3.Loading Data

- The dataset has been collected from the **analyticsvidhya.com and the dataset in **CSV format**.
- To load the data pd.read_csv("data//path").
- From the data 614 rows and 13 features are gained.

## 4. Data description and acquistititon

| Variable | Description |
|---|---|
|Loan_ID|Unique Loan ID|
|Gender|Male/ Female|
|Married|Applicant married (Y/N)|
|Dependents|Number of dependents|
|Education|Applicant Education (Graduate/ Under Graduate)|
|Self_Employed|Self employed (Y/N)|
|ApplicantIncome|Applicant income|
|CoapplicantIncome|Coapplicant income|
|LoanAmount|Loan amount in thousands|
|Loan_Amount_Term|Term of loan in months|
|Credit_History|credit history meets guidelines|
|Property_Area|Urban/ Semi Urban/ Rural|
|Loan_Status(Target)| Loan approved (Y/N|
