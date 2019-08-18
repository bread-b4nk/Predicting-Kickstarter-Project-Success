# Kickstarter Project

## Things learnt
   - a rough idea of how many classification algorithms work: decision trees, random forest, boosting classifier, naive bayes, ADABoost, support vector classifiers
   - Pandas python library
   - Numpy python library
   - SKLearn python library
   - using WEKA software
   - preprocessing required before any training e.g. one hot encoding
   - datetime python library, finding the time between two dates
   - using Jupyter Notebook
## Details

This project was started on the 6th of July, 2018. I started learning about Data Science and how to apply it using python on this [udemy course](https://www.udemy.com/python-for-data-science-and-machine-learning-bootcamp/learn/v4/overview). With no previous knowledge on Data Science, I expect this project to have a lot of room for improvement.

## Introduction

With nearly 15 million users funding over $3.7 billion for others, Kickstarter, a Benefit Corporation since 2015, has proven to be a well known and loved by the internet community. It has fulfilled the dreams of many, successfully funding over 146,000 different projects. Kickstarter themselves have already done a bit of research into the projects on their website, looking at the amount of money made for both successful and unsuccessful projects; their results show, however, that only 36% of projects yield success.

## Objective

This study aims to be able to predict whether a kickstarter projectwould be successful or not by looking at multiple different variables and their relationship with a projects’ success.

## Materials & Methods

All of the coding was done in Python in the Jupyter Notebook. The dataset containing all the information on Kickstarter was from [Kaggle](https://www.kaggle.com/kemical/kickstarter-projects )

```
Attribute			Description
ID				    the ID of the project
name				the name of the project
pledged			    sum of money backers “pledged” to donate
backers			    number of people supporting the project (numeric)
state				whether the project was successful or not (binary: successful or failed)
main_category		generalised type of project (nominal)
currency			type of currency used for project (nominal)
country		    	country of origin of project (nominal)
deadline			date for when the project must be funded by (YYYY-MM-DD)
goal				set amount of money required for a successful project (numeric)
launched			date for when the project was launched (YYYY-MM-DD)
```

### Acquiring the Data

```
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

%matplotlib inline

ks_project = pd.read_csv("ks-projects-201801.csv")
```

### Cleaning and Analyzing the Data

The data that wasn't going to be used was first deleted:

```
ks_project.drop("category",axis = 1, inplace = True)
ks_project.drop("ID",axis = 1, inplace= True)
ks_project.drop("name", axis = 1, inplace = True)
ks_project.drop("pledged", axis = 1, inplace = True)
ks_project.drop("usd_pledged_real", axis = 1, inplace = True)
ks_project.drop("usd pledged", axis = 1, inplace = True)
ks_project.drop("backers",axis=1,inplace = True)
ks_project.drop("goal",axis=1,inplace = True)
```
**_State_**

There were many possible states for all the kickstarter projects: successful, failed, cancelled, undefined, live or suspended. All the projects that were “successful” were changed to 1, all that were “cancelled” were changed the 0 to create a binary column; the rows with the other states were removed from the dataset. This is because with the objective of being able to predict whether a project would be successful or not, projects that are still live won’t provide the study good data since the result of the project isn’t known yet, furthermore projects that were cancelled, undefined or suspended would have been caused by other factors uncontained in the provided dataset. Here is how the states were changed:

```
cleaned_state = {"failed":0,"canceled":np.nan,"successful":1,"undefined":np.nan,"live":np.nan,"suspended":np.nan}
ks_project["state"] = ks_project["state"].map(cleaned_state)
ks_project = ks_project.dropna(axis=0)
```

**_Categories_**

The individual categories were too general and extensive to be divided or simplified as there were so many categories, so this study looked at “Main Category” instead, which generalised all the categories in to 15 categories: Publishing, Film & Video, Music, Food, Crafts, Games, Design, Comics, Fashion, Theater, Art, Photography, Technology, Dance, and Journalism. Not many changes had to really be made to it. When graphed, the worst category was Technology with a 24% chance of being successful. This is how it was graphed:

```
category_graph = ks_project[["main_category","state"]].groupby(["main_category"], as_index=False).mean()
sns.factorplot(x = "main_category",y = "state", data = category_graph, kind = "bar", size = 5, aspect = 3)
```

**_Currency_**

Similarly to the categories, different currencies were also graphed against the percentage of success to look for any anomalies or prominence; Hong Kong was the most successful currency at a 45% of success, with the worst being Euros at 27%. This was how I found the average success rate and graphed it:

```
currency_graph = ks_project[["currency","state"]].groupby(["currency"]), as_index=False).mean()
sns.factorplot(x = "currency",y = "state", data = currency_graph, kind = "bar", size = 5, aspect = 3)
````

**_Country_**

The countries involved were also graphed against each other using the code:

```
country_graph = ks_project[["country","state"]].groupby(["country"], as_index=False).mean()
sns.factorplot(x = "country",y = "state", data = country_graph, kind = "bar", size = 5, aspect = 3)
```

**_Goal_**

The column “usd_goal_real”, which is the amount of money in dollars that a project required to deem itself “successful”, was separated into 5 evenly numbered ranges. Then a trend was searched for between the ranges of goals and the average chance of success per range:

```
ks_project["goal_ranges"] = pd.qcut(ks_project["usd_goal_real"],5)
sns.factorplot(x="goal_ranges",y="state",data=ks_project,size=4,aspect=3)
```

**_Launched and Deadline–Time Spent_**

Being provided with “launched” and “deadline” a new column was created called “time_spent” which contains the date in “deadline” subtracted by “launched” to calculate how much time (in days) was actually spent for each project. Then the time_spent was separated into 3 evenly numbered ranges to be graphed against the chances of average chance of success for each range.

```
from datetime import timedelta
import datetime

datetimefmt = "%Y-%m-%d"
datetimefmt2 = "%Y-%m-%d %H:%M:%S"
time_spent = []

for i in range(len(ks_project.index)):

    dt1 = datetime.datetime.strptime(ks_project["deadline"][i],datetimefmt)
    dt2 = datetime.datetime.strptime(ks_project["launched"][i],datetimefmt2)
    ans = dt1.date()- dt2.date()
    time_spent.append(ans.days)

ks_project["time_spent"] = time_spent
```

### Setting up training and test data

This code doesn't include the cross validation done on the Weka application afterwards, but this was how I originally prepared the training and test data

```
from sklearn.model_selection import train_test_split

y = ks_project["state"]
X = ks_project.drop("state",axis=1)
```

Many different models were then tested, here's the coding for each of them:

**_Decision Trees_**

```
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)

dtree_pred = dtree.predict(X_test)
print(classification_report(y_test,dtree_pred))
```

**_Random Forests_**

```
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train,y_train)

rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))
```

**_Gradient Boosting Classifier_**

```
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier(n_estimators=100,max_depth=1,random_state=0)
gbc.fit(X_train,y_train)

gbc_pred = gbc.predict(X_test)
print(classification_report(y_test,gbc_pred))
```

**_MLP Classifier_**

```
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver='lbfgs',alpha=1e-5,hidden_layer_sizes=(21, 2),random_state=1)

mlp.fit(X_train, y_train)

mlp_pred = mlp.predict(X_test)
print(classification_report(y_test,mlp_pred))
```

**_AdaBoost Classifier_**

```
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),algorithm="SAMME",n_estimators=200)

abc.fit(X_train, y_train)

abc_pred = abc.predict(X_test)
print(classification_report(y_test,abc_pred))
```

**_Naive Bayes_**

```
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.3,max_features=0.3)

bagc = bagging.fit(X_train,y_train)

bagc_pred = bagc.predict(X_test)
print(classification_report(y_test,bagc_pred))
```

**_SVC_**

```
from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(X_train,y_train)

gnb_pred = gnb.predict(X_test)
print(classification_report(y_test,gnb_pred))
```

### Preprocessing

**_One Hot Encoding_**

I'm not completely sure about the indepth mathematical reasoning behind this, but I can see why it would help increase the accuracy of models;  it would probably be better if I don't put the code on this page as it's a huge chunk of data,it's on the Kickstarter Project python file. Here's [more information]                                          (https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) on one hot encoding.

## Results

**_Results from Jupyter Notebook_**

Note that these results did not go through cross validation, these were the very first results I recieved so the scores are pretty low:

```
                        Model  Score
0               Decision Tree  62.78
1               Random Forest  62.87
2  GradientBoostingClassifier  63.70
3              MLP Classifier  60.35
4          AdaBoostClassifier  64.24
5           BaggingClassifier  62.00
6                 Naive Bayes  59.00
7   Support Vector Classifier  63.42
```

**_Results from WEKA_**

[WEKA](https://www.cs.waikato.ac.nz/ml/weka/) was also used because it could perform [cross validation](https://www.openml.org/a/estimation-procedures/1) more easily, I did only 5 folds to see if the accuracy would increase at all. However my computer was too slow to run Random Forest, SVC, and GradientBoosting Classifier couldn't be found on the software so it's missing three models:

```
                        Model   Score
0               Decision Tree   63.59
1               Random Forest   NaN     
2  GradientBoostingClassifier   NaN
3              MLP Classifier   63.10
4          AdaBoostClassifier   64.68
5           BaggingClassifier   63.71
6                 Naive Bayes   53.06
7   Support Vector Classifier   NaN
```

Looking at the results most of them improved except for Naive Bayes.
