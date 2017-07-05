import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('/home/Akai/Downloads/Loan prediction/train.csv')

data = data.drop(labels = ['Loan_ID'], axis = 1)
data = data.fillna(method = 'ffill', axis = 0)
data = data.fillna(method = 'bfill', axis = 0)

data.ApplicantIncome = data.ApplicantIncome.astype(float)

#converting variables to dummies
ordered_gender = ['Male', 'Female']
data.Gender = data.Gender.astype("category", ordered=True, categories=ordered_gender).cat.codes

ordered_Married = ['No', 'Yes']
data.Married = data.Married.astype("category", ordered=True, categories=ordered_Married).cat.codes

ordered_Education = ['Not Graduate', 'Graduate']
data.Education = data.Education.astype("category", ordered=True, categories=ordered_Education).cat.codes

ordered_Self = ['No', 'Yes']
data.Self_Employed = data.Self_Employed.astype("category", ordered=True, categories=ordered_Self).cat.codes

ordered_loan = ['N', 'Y']
data.Loan_Status = data.Loan_Status.astype("category", ordered=True, categories=ordered_loan).cat.codes

for i in range(len(data)):
    if data.Dependents[i] == '0':
        data.Dependents[i] = 0
    elif data.Dependents[i] == '1':
        data.Dependents[i] = 0.5
    else:
        data.Dependents[i] = 1

for i in range(len(data)):    
    if data.Property_Area[i] == 'Urban':
        data.Property_Area[i] = 0
    else:
        data.Property_Area[i] = 1
        
app_inc_max = np.max(data.ApplicantIncome)
app_inc_min = np.min(data.ApplicantIncome)
co_app_inc_max = np.max(data.CoapplicantIncome)
co_app_inc_min = np.min(data.CoapplicantIncome)
loan_amt_max = np.max(data.LoanAmount)
loan_amt_min = np.min(data.LoanAmount)
loan_amt_term_max = np.max(data.Loan_Amount_Term)
loan_amt_term_min = np.min(data.Loan_Amount_Term)

for i in range(len(data)):
    data.ApplicantIncome[i] = ((data.ApplicantIncome[i] - app_inc_min)/(app_inc_max-app_inc_min))
    data.CoapplicantIncome[i] = ((data.CoapplicantIncome[i] - co_app_inc_min)/(co_app_inc_max - co_app_inc_min))
    data.LoanAmount[i] = ((data.LoanAmount[i]-loan_amt_min)/(loan_amt_max-loan_amt_min))
    data.Loan_Amount_Term[i] = ((data.Loan_Amount_Term[i]-loan_amt_term_min)/(loan_amt_term_max - loan_amt_term_min))
    


target = pd.DataFrame({'Loan_Status':data.Loan_Status})
data = data.drop(labels = ['Loan_Status'], axis = 1)



from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier(loss = 'exponential', n_estimators = 100 , learning_rate = 0.05, random_state = 0)

data1 = data.drop(labels = ['Loan_Amount_Term'], axis = 1)
model = gbm.fit(data1, target)
from sklearn.cross_validation import cross_val_score

score = cross_val_score(model, data1, target['Loan_Status'].copy(), cv = 10)
from scipy.stats import sem
def mean_scores(scores):
    return("Mean score: {0:.3f}(+/-{1:.3f})").format(np.mean(scores), sem(scores))
    
print(mean_scores(score))


test = pd.read_csv('/home/Akai/Downloads/Loan prediction/test.csv')
Id = list()
for i in range(len(test)):
    Id.append(test['Loan_ID'][i])
    
test = test.drop(labels = ['Loan_ID'], axis =1)
test = test.fillna(method = 'ffill', axis = 0)
test = test.fillna(method = 'bfill', axis = 0)

test.ApplicantIncome = test.ApplicantIncome.astype(float)
test.CoapplicantIncome = test.CoapplicantIncome.astype(float)


test.Gender = test.Gender.astype("category", ordered=True, categories=ordered_gender).cat.codes
test.Married = test.Married.astype("category", ordered=True, categories=ordered_Married).cat.codes
test.Education = test.Education.astype("category", ordered=True, categories=ordered_Education).cat.codes
test.Self_Employed = test.Self_Employed.astype("category", ordered=True, categories=ordered_Self).cat.codes


for i in range(len(test)):
    if test.Dependents[i] == '0':
        test.Dependents[i] = 0
    elif test.Dependents[i] == '1':
        test.Dependents[i] = 0.5
    else:
        test.Dependents[i] = 1

for i in range(len(test)):    
    if test.Property_Area[i] == 'Urban':
        test.Property_Area[i] = 0
    else:
        test.Property_Area[i] = 1
        
for i in range(len(test)):
    test.ApplicantIncome[i] = ((test.ApplicantIncome[i] - app_inc_min)/(app_inc_max-app_inc_min))
    test.CoapplicantIncome[i] = ((test.CoapplicantIncome[i] - co_app_inc_min)/(co_app_inc_max - co_app_inc_min))
    test.LoanAmount[i] = ((test.LoanAmount[i]-loan_amt_min)/(loan_amt_max-loan_amt_min))
    test.Loan_Amount_Term[i] = ((test.Loan_Amount_Term[i]-loan_amt_term_min)/(loan_amt_term_max - loan_amt_term_min))

test = test.drop(labels = ['Loan_Amount_Term'], axis = 1)
predictions = gbm.predict(test)
pred = []
for i in range(len(predictions)):
    pred.append(predictions[i])
    
for i in range(len(pred)):
    if pred[i] == 0:
        pred[i] = 'N'
    else:
        pred[i] = 'Y'

submission = pd.DataFrame({'Loan_ID':Id, 'Loan_Status':pred})
submission.to_csv('submission.csv')