# GFC Credit Score Classification 

## Problem description 
Over the years, the GFC (Global Finance Company) has collected basic bank details and gathered a lot of credit-related information. The management wants to build an intelligent system to segregate the people into credit score brackets to reduce the manual efforts. The objective is to build a machine learning model that can classify the credit score based in a person’s credit-related information, into 3 brackets: Poor, Standard and Good Credit Score.

Classifying effectively each person credit score helps the GFC to create a profile for that particular person that would help in the decision taking for interest rates, loans and negotiation with that person. Besides being a good indicator for future Machine Learning problems such as risk perception.

## Decision Making process

With the training data, It was noticeable that the target (Credit_Score) was unbalanced showing more observations labeled as "Standard". From the beginning I knew that my biggest challenge through this project was to create a Machine Learning Model that can differentiate between the unbalanced labels (Poor and Good credit score).

![Target balance](https://raw.githubusercontent.com/liamarguedas/data/main/Data-Science/Credit-Score-Classification/Summary-Charts/Target%20Balace.png)

In order to create a model that can differentiate between the unbalanced labels I needed to understand the profile of each one (Poor and Good credit score), understanding those patterns and looking for them in the data will made a good performance model for the problem, the statistics of each profile showed me the following:

![Target Statistics](https://raw.githubusercontent.com/liamarguedas/data/main/Data-Science/Credit-Score-Classification/Summary-Charts/Target%20Statistics.png)

It is noticeable that there were manny cut off points to determine if someone has good or poor credit score, the ones I notice are:

- Older people usually have good credit score than youger people, the median showed Good credit score age as 36 and Poor credit score as 31.

- People with Good credit score have higher income than people with Poor credit score as showed in the columns "Annual_Income" and "Monthly_Inhand_Salary"

- People with Good credit score usually have less bank accounts that people with Poor credit score, the meadian in this case is 3 for the Good credit score and 5 for the Poor credit score

- People with Good credit score usually have better interest rates than people with Poor credit score.

- People with Good credit score have less loans than people with Poor credit score.

By that point I knew that I have the data I needed to create a good model. There was a few challenges through the project like many arbitrary values and the data that needed to be filled, also there were many outliers in some feature that needed to be understand and decide if they were worth removing or leaving. Some outliers showed good patterns for our model, such as Total_EMI_per_month:

![EMI_per_month](https://raw.githubusercontent.com/liamarguedas/data/main/Data-Science/Credit-Score-Classification/Summary-Charts/EMI%20per%20Month.png)

Outliers than ended up being valuable for the models performance. Since it was a multi-class classification problem I needed to choose Machine Learning algorithms that support this type of classification, I decided to try with:

- K-Nearest Neighbors
- Random Forest
- Gradient Boosting
- Keras NN

For classifications problems I could not based my solution only in the Score metric so I decided to use a ROC AUC, Accuracy, Cohen’s Kappa, Recall, Precision and F1-score. I also needed a baseline score to understand if my model was actually working or it was only making random predictions, my baseline metrics were calulate by:

- ZeroR value: ZeroR score is the score I will get if I predict everything as zero, this means if the model only predicts Standard credit scores, in order to get a good model I will have to get an score higher than that.

    ZeroR score: 0.5493287321546753

- Random Rate Classifier (Weighted Guessing): Calculated by squaring our classed percentage and sum all the values.

    Random Rate score: 0.55 ** 2 + 0.26 ** 2 + 0.19 ** 2 = 0.40620000000000006

Knowing what performance the model needed to achieve in order to be useful I created a cross validation scaling the data in each fold to reduce data leakage and I managed to score the next results:

![Algorithms Performance](https://raw.githubusercontent.com/liamarguedas/data/main/Data-Science/Credit-Score-Classification/Summary-Charts/Algorithms%20Performance.png)

## Final Decision

Random Forest showed the best performance out of all of the algorithms So I decided to use Random Forest and tune its hyperparameters, after the tuned I trained another model in a cross validation to see its performance and got the following results:

![Random Forest Performance](https://raw.githubusercontent.com/liamarguedas/data/main/Data-Science/Credit-Score-Classification/Summary-Charts/Random%20Forest%20Classification%20Report.png)

The model got a precision of 0.75 in classifying 1 (Good credit score) and 0.78 in classifying 2 (Poor credit score) and an F1-score (It describes the optimal blend of precision and recall combined) of 0.74 for 1 (Good credit score) and 0.79 for 2 (Poor credit score) with an overall accuracy (Including 0 Standard) of 0.80.

The confuision matrix for this tunned model were showing the next results:

![Confusion Matrix](https://raw.githubusercontent.com/liamarguedas/data/main/Data-Science/Credit-Score-Classification/Summary-Charts/Random%20Forest%20Confusion%20Matrix.png)

It is noticeable that the model really learned to differentiate between 1 (Good credit score) and 2 (Poor credit score) which was the objetive. From the confusion matrix out of 17.000 there is only 25 observations predicted as 2 (Poor credit score) when they were really 1 (Good credit score) and 70 observatios predicted as 1 (Good credit score) when they were really 2 (Poor credit score). This means that from the cross validation the model performed very well classifying the data points.

Since Random Forest showed an amazing performance I decided to train the final model with the entire dataset (85.000 Observations) in order to solve the problem and understand the importance of each feature trainned. The feature importance graph looked like:

![Feature importance](https://raw.githubusercontent.com/liamarguedas/data/main/Data-Science/Credit-Score-Classification/Summary-Charts/Random%20Forest%20Feature%20Importance.png)

From the chart we can see that they were a lot of features that were not that much important for the model and among those that were important in the trainning we have "Outstanding_Debt", "Interest_Rate" and "Credit_History_Age". Since the GFC provided a testing data, I predicted the values of the data and compare their statistics to the original (Model trained) data set to see if the Random Forest model could really describe our data.

![Statistics](https://raw.githubusercontent.com/liamarguedas/data/main/Data-Science/Credit-Score-Classification/Summary-Charts/Train%20vs%20Test%20Statistics.png)

We can see that the model predictions hold the statistics from the original data set, which is good for our problem. An example of this is age, at the beginning I said that older people usually have good credit score than younger people and the predicted variables showed the same behavior and the patterns hold for each continuos feature in the project. 

## Did my decision solve the GFC problem?

From the final statistics is noticeable that the model predictions can really describe the data with a really good precision, since it is an unbalanced target there's gonna be some datapoints that were predicted as 0 (Standard) but I managed to reduced the amount of observations that were 1 (Good) and got predicted 2 (Poor) and vice-versa. Which was the main objective of the project, since predicting 1 (Good) and 2 (Poor) as 0 (Standard) was not a such a deal for the GFC, differentiate between Good and Poor credit score was the main goal. It would be a huge mistake and problem if the model predicted someone with poor credit score as having a good credit score and I worked focusing in minimazing those predictions and erros. I am satisfied with the results delivered and it satisfies the company goals. So in overall it solved the GFC Problem.

## Is there any room for improvement?

I tried training the model with and without engineered features and I found no difference. I also tried trainning the model without the features with little to no correlation and without outliers and there was no difference, the best performance model without overfitting was the one I showcased in the project together with the hyperparameters tunned, there could be room for improvement in the creation of new feature engineered variables and transformations (that might affect the model in production) or new valuable features to trained the model again.

Random Forest with the hyperparameters tunned showed a model that describes the data with a good precision and can differentiate between the classes, having a low cost of implementation and a sustainable model overall. I have managed to solve the entire problem, developed a really good approach and an amazing data science project.