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

- Total **7** Object datatype and **6** Numerical datatype are recorded.
- The average Applicant Income and Loan Amount are **$5403.3** and **$146**.

# 5. Data Visualization

<p align="center">
     <img width="400" height="200" src="https://user-images.githubusercontent.com/119164734/209465592-8636a407-e176-40b1-9509-18be70578592.jpg">
</p>

EDA is applied **to investigate the data and summarize the key insights**. It will give you the basic understanding of your data, it's distribution, null values and much more. You can either explore data using graphs or through some python functions.

### Observation:-

- From the categorical features, **Male** has the highest contribution for Loan aprroval duties.
- Employees has the highest aspire than Self-employees and Semi urban has the huge contribution in Loan approval records.

# 7. Data Postprocessing

## Encoding Categorical Variable
- Encoding is a technique of converting categorical variables into numerical values so that it could be easily fitted to a machine learning model.
I have used the Label / Ordinal encoding.

**from sklearn.preprocessing import LabelEncoder**

**Label=LabelEncoder()**

**data_1["Fuel Type"]=Label.fit_transform(data_1["Fuel Type"])**

- More than 5 features has been changed to Numerical features. 

# 8.Separating train and test data:

- The train-test split is used to estimate the performance of machine learning algorithms that are applicable for prediction-based Algorithms/Applications. This method is a fast and easy procedure to perform such that we can compare our own machine learning model results to machine results.

**from sklearn.model_selection import train_test_split**

**X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=45)**

<p align="center">
  <img width="400" height="200" src="https://user-images.githubusercontent.com/119164734/208979910-33a3af21-734b-4ef9-8cc3-b4c51efed7af.png">
</p>

- The dataset has been separated into 70% train data and 30% test data.


# 9.Modeling the Train_data :-

Define a baseline model:

A baseline model is essentially a simple model that acts as a reference in a machine learning project. Its main function is to contextualize the results of trained models. Baseline models usually lack complexity and may have little predictive power.

### i) KNeighborsClassifier Model

<p align="center">
  <img width="400" height="200" src="https://user-images.githubusercontent.com/119164734/209466406-12e21731-ce4f-4fe8-bed3-7191d0303bb3.png">
</p>

- k' in KNN is a parameter that refers to **the number of nearest neighbours to include in the majority of the voting process**.
- KNN algorithm can be used for both classification and regression problems. 

### ii) RandomForest Classifier

<p align="center">
  <img width="400" height="200" src="https://user-images.githubusercontent.com/119164734/209467144-9084dfb4-dc00-4955-b360-47395115a4c5.jpeg">
</p>

**Random forest classifier creates a set of decision trees from randomly selected subset of training set**. It then **aggregates the votes from different decision trees to decide the final class of the test object**.

Among all the available classification methods, random forests provide the **highest accuracy** and it can automatically balance data sets when a class is more infrequent than other classes in the data.

### iii) Support Vector Classifier

<p align="center">
  <img width="400" height="200" src="https://user-images.githubusercontent.com/119164734/209468133-ba33c306-7ebd-4761-99dd-27c3749963e6.png"
</p>

- SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a **hyperplane**.

- SVM chooses the extreme points/vectors that help in creating the hyperplane. These extreme cases are called as support vectors, and hence algorithm is termed as Support Vector Machine. 
     
- For a best margin, we can use a **kernel**,**Regularization** and **margin** as parameter.
     
### iv) LogisticRegression Classifier
     
  <p align="center">
  <img width="400" height="200" src="https://user-images.githubusercontent.com/119164734/209468545-46153968-d5ef-44eb-8b7d-c495c2d53960.png"
</p>
   
- Logistic Regression is used to predict categorical variables with the help of dependent variables. Consider there are two classes and a new data point is to be checked which class it would belong to. Then algorithms compute probability values that range from 0 and 1
       
-  This model process on **Threshold Technique**, so above the 0.5 is mention as 1 and below value is represent as 0.
       
       
#Observation:-
- **X_train and Y_train data** has been fitted in this four models to find out the low Overfitting model
- **Logistic Regression Classifier** is the best accuracy scorer among the models.It has 82% of accuracy in train data and 77% of accuracy in test data.
- **RandomforestClassifier** gives the best accuracy score and Low Basis and low variance (low overfitting) among the models.So the RandomForestClassifier is used for the **Hyperparameter tuning**.
- **KneighborsClassifier** and **RandomForest classifier** has the Low Basis and High variance(Overfitting).

