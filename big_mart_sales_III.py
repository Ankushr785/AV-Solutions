import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


sale_data = pd.read_csv('/home/Akai/Downloads/Big Mart III/Train.csv')
test = pd.read_csv('/home/Akai/Downloads/Big Mart III/Test.csv')

for i in range(len(sale_data)):
    sale_data.Outlet_Identifier[i] = sale_data.Outlet_Identifier[i][-3:]
    
sale_data.Outlet_Identifier = sale_data.Outlet_Identifier.astype(int)

for i in range(len(test)):
    test.Outlet_Identifier[i] = test.Outlet_Identifier[i][-3:]
    
test.Outlet_Identifier = test.Outlet_Identifier.astype(int)
unique = sale_data.Outlet_Identifier.unique()

uniqueness = []
for i in range(len(unique)):
    uniqueness.append(unique[i])
    
prediction_list = []

data = sale_data[(sale_data.Outlet_Identifier == 10)].reset_index(drop = True)
needy_data = pd.DataFrame({'Item_Type':data.Item_Type, 'Item_MRP':data.Item_MRP})
target = pd.DataFrame({'Item_Outlet_Sales':data.Item_Outlet_Sales})
    
needy_data = needy_data.fillna(method = 'bfill', axis = 0)
needy_data = needy_data.fillna(method = 'ffill', axis = 0)
needy_data = needy_data.fillna(0)
   
target = target.fillna(method = 'bfill', axis = 0)
target = target.fillna(method = 'ffill', axis = 0)
target = target.fillna(0)

ordered_type = []
for k in range(len(needy_data.Item_Type.unique())):
    ordered_type.append(needy_data.Item_Type.unique()[k])

needy_data.Item_Type = needy_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
needy_data.Item_Type = needy_data.Item_Type.astype(float)

it_t_min = np.min(needy_data.Item_Type)
it_t_max = np.max(needy_data.Item_Type)
#it_wt_max = np.max(needy_data.Item_Weight)
#it_wt_min = np.min(needy_data.Item_Weight)
it_p_max = np.max(needy_data.Item_MRP)
it_p_min = np.min(needy_data.Item_MRP)
it_s_min = np.min(target.Item_Outlet_Sales)
it_s_max = np.max(target.Item_Outlet_Sales)

for l in range(len(needy_data)):
    needy_data.Item_Type[l] = ((needy_data.Item_Type[l]-it_t_min)/(it_t_max - it_t_min))
    #needy_data.Item_Weight[l] = ((needy_data.Item_Weight[l] - it_wt_min)/(it_wt_max - it_wt_min))
    needy_data.Item_MRP[l] = ((needy_data.Item_MRP[l] - it_p_min)/(it_p_max - it_p_min))
    target.Item_Outlet_Sales[l] = ((target.Item_Outlet_Sales[l] - it_s_min)/(it_s_max - it_s_min))

from sklearn.ensemble import GradientBoostingRegressor
gbm = GradientBoostingRegressor(loss = 'ls', n_estimators = 50, learning_rate = 0.05, random_state = 0)

model = gbm.fit(needy_data, target)
from sklearn.cross_validation import cross_val_score

score = cross_val_score(model, needy_data, target['Item_Outlet_Sales'].copy(), cv = 10)
from scipy.stats import sem
def mean_scores(scores):
    return("Mean score: {0:.3f}(+/-{1:.3f})").format(np.mean(scores), sem(scores))
    
print(mean_scores(score))
    
t_data = test[(test.Outlet_Identifier == 10)].reset_index(drop = True)

test_data = pd.DataFrame({'Item_Type':t_data.Item_Type, 'Item_MRP':t_data.Item_MRP})


ordered_type = []
for q in range(len(test_data.Item_Type.unique())):
    ordered_type.append(test_data.Item_Type.unique()[q])

test_data.Item_Type = test_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
test_data.Item_Type = test_data.Item_Type.astype(float)

for r in range(len(test_data)):
    test_data.Item_Type[r] = ((test_data.Item_Type[r]-it_t_min)/(it_t_max - it_t_min))
    test_data.Item_MRP[r] = ((test_data.Item_MRP[r] - it_p_min)/(it_p_max - it_p_min))
    #test_data.Item_Weight[r] = ((test_data.Item_Weight[r] - it_wt_min)/(it_wt_max - it_wt_min))
    
pred = []
predictions = gbm.predict(test_data)
for z in range(len(predictions)):
    pred.append(predictions[z])
        
for y in range(len(pred)):
    pred[y] = (pred[y]*(it_s_max-it_s_min)) + (it_s_min)
        
for x in range(len(pred)):
    prediction_list.append(pred[x])
    

data = sale_data[(sale_data.Outlet_Identifier == 13)].reset_index(drop = True)
needy_data = pd.DataFrame({'Item_Weight':data.Item_Weight, 'Item_Type':data.Item_Type, 'Item_MRP':data.Item_MRP})
target = pd.DataFrame({'Item_Outlet_Sales':data.Item_Outlet_Sales})
    
needy_data = needy_data.fillna(method = 'bfill', axis = 0)
needy_data = needy_data.fillna(method = 'ffill', axis = 0)
needy_data = needy_data.fillna(0)
   
target = target.fillna(method = 'bfill', axis = 0)
target = target.fillna(method = 'ffill', axis = 0)
target = target.fillna(0)

ordered_type = []
for k in range(len(needy_data.Item_Type.unique())):
    ordered_type.append(needy_data.Item_Type.unique()[k])

needy_data.Item_Type = needy_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
needy_data.Item_Type = needy_data.Item_Type.astype(float)

it_t_min = np.min(needy_data.Item_Type)
it_t_max = np.max(needy_data.Item_Type)
it_wt_max = np.max(needy_data.Item_Weight)
it_wt_min = np.min(needy_data.Item_Weight)
it_p_max = np.max(needy_data.Item_MRP)
it_p_min = np.min(needy_data.Item_MRP)
it_s_min = np.min(target.Item_Outlet_Sales)
it_s_max = np.max(target.Item_Outlet_Sales)

for l in range(len(needy_data)):
    needy_data.Item_Type[l] = ((needy_data.Item_Type[l]-it_t_min)/(it_t_max - it_t_min))
    needy_data.Item_Weight[l] = ((needy_data.Item_Weight[l] - it_wt_min)/(it_wt_max - it_wt_min))
    needy_data.Item_MRP[l] = ((needy_data.Item_MRP[l] - it_p_min)/(it_p_max - it_p_min))
    target.Item_Outlet_Sales[l] = ((target.Item_Outlet_Sales[l] - it_s_min)/(it_s_max - it_s_min))


gbm = GradientBoostingRegressor(loss = 'lad', n_estimators = 70, learning_rate = 0.05, random_state = 0)

model = gbm.fit(needy_data, target)


score = cross_val_score(model, needy_data, target['Item_Outlet_Sales'].copy(), cv = 10)

def mean_scores(scores):
    return("Mean score: {0:.3f}(+/-{1:.3f})").format(np.mean(scores), sem(scores))
    
print(mean_scores(score))
    
t_data = test[(test.Outlet_Identifier == 13)].reset_index(drop = True)

test_data = pd.DataFrame({'Item_Weight':t_data.Item_Weight, 'Item_Type':t_data.Item_Type, 'Item_MRP':t_data.Item_MRP})


ordered_type = []
for q in range(len(test_data.Item_Type.unique())):
    ordered_type.append(test_data.Item_Type.unique()[q])

test_data.Item_Type = test_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
test_data.Item_Type = test_data.Item_Type.astype(float)

for r in range(len(test_data)):
    test_data.Item_Type[r] = ((test_data.Item_Type[r]-it_t_min)/(it_t_max - it_t_min))
    test_data.Item_MRP[r] = ((test_data.Item_MRP[r] - it_p_min)/(it_p_max - it_p_min))
    test_data.Item_Weight[r] = ((test_data.Item_Weight[r] - it_wt_min)/(it_wt_max - it_wt_min))
    
pred = []
predictions = gbm.predict(test_data)
for z in range(len(predictions)):
    pred.append(predictions[z])
        
for y in range(len(pred)):
    pred[y] = (pred[y]*(it_s_max-it_s_min)) + (it_s_min)
        
for x in range(len(pred)):
    prediction_list.append(pred[x])
    
    
data = sale_data[(sale_data.Outlet_Identifier == 17)].reset_index(drop = True)
needy_data = pd.DataFrame({'Item_Weight':data.Item_Weight, 'Item_Type':data.Item_Type, 'Item_MRP':data.Item_MRP})
target = pd.DataFrame({'Item_Outlet_Sales':data.Item_Outlet_Sales})
    
needy_data = needy_data.fillna(method = 'bfill', axis = 0)
needy_data = needy_data.fillna(method = 'ffill', axis = 0)
needy_data = needy_data.fillna(0)
   
target = target.fillna(method = 'bfill', axis = 0)
target = target.fillna(method = 'ffill', axis = 0)
target = target.fillna(0)

ordered_type = []
for k in range(len(needy_data.Item_Type.unique())):
    ordered_type.append(needy_data.Item_Type.unique()[k])

needy_data.Item_Type = needy_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
needy_data.Item_Type = needy_data.Item_Type.astype(float)

it_t_min = np.min(needy_data.Item_Type)
it_t_max = np.max(needy_data.Item_Type)
it_wt_max = np.max(needy_data.Item_Weight)
it_wt_min = np.min(needy_data.Item_Weight)
it_p_max = np.max(needy_data.Item_MRP)
it_p_min = np.min(needy_data.Item_MRP)
it_s_min = np.min(target.Item_Outlet_Sales)
it_s_max = np.max(target.Item_Outlet_Sales)

for l in range(len(needy_data)):
    needy_data.Item_Type[l] = ((needy_data.Item_Type[l]-it_t_min)/(it_t_max - it_t_min))
    needy_data.Item_Weight[l] = ((needy_data.Item_Weight[l] - it_wt_min)/(it_wt_max - it_wt_min))
    needy_data.Item_MRP[l] = ((needy_data.Item_MRP[l] - it_p_min)/(it_p_max - it_p_min))
    target.Item_Outlet_Sales[l] = ((target.Item_Outlet_Sales[l] - it_s_min)/(it_s_max - it_s_min))


gbm = GradientBoostingRegressor(loss = 'huber', n_estimators = 70, learning_rate = 0.05, random_state = 0)

model = gbm.fit(needy_data, target)


score = cross_val_score(model, needy_data, target['Item_Outlet_Sales'].copy(), cv = 10)

def mean_scores(scores):
    return("Mean score: {0:.3f}(+/-{1:.3f})").format(np.mean(scores), sem(scores))
    
print(mean_scores(score))
    
t_data = test[(test.Outlet_Identifier == 17)].reset_index(drop = True)

test_data = pd.DataFrame({'Item_Weight':t_data.Item_Weight, 'Item_Type':t_data.Item_Type, 'Item_MRP':t_data.Item_MRP})


ordered_type = []
for q in range(len(test_data.Item_Type.unique())):
    ordered_type.append(test_data.Item_Type.unique()[q])

test_data.Item_Type = test_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
test_data.Item_Type = test_data.Item_Type.astype(float)

for r in range(len(test_data)):
    test_data.Item_Type[r] = ((test_data.Item_Type[r]-it_t_min)/(it_t_max - it_t_min))
    test_data.Item_MRP[r] = ((test_data.Item_MRP[r] - it_p_min)/(it_p_max - it_p_min))
    test_data.Item_Weight[r] = ((test_data.Item_Weight[r] - it_wt_min)/(it_wt_max - it_wt_min))
    
pred = []
predictions = gbm.predict(test_data)
for z in range(len(predictions)):
    pred.append(predictions[z])
        
for y in range(len(pred)):
    pred[y] = (pred[y]*(it_s_max-it_s_min)) + (it_s_min)
        
for x in range(len(pred)):
    prediction_list.append(pred[x])
    
    
data = sale_data[(sale_data.Outlet_Identifier == 18)].reset_index(drop = True)
needy_data = pd.DataFrame({'Item_Type':data.Item_Type, 'Item_MRP':data.Item_MRP})
target = pd.DataFrame({'Item_Outlet_Sales':data.Item_Outlet_Sales})
    
needy_data = needy_data.fillna(method = 'bfill', axis = 0)
needy_data = needy_data.fillna(method = 'ffill', axis = 0)
needy_data = needy_data.fillna(0)
   
target = target.fillna(method = 'bfill', axis = 0)
target = target.fillna(method = 'ffill', axis = 0)
target = target.fillna(0)

ordered_type = []
for k in range(len(needy_data.Item_Type.unique())):
    ordered_type.append(needy_data.Item_Type.unique()[k])

needy_data.Item_Type = needy_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
needy_data.Item_Type = needy_data.Item_Type.astype(float)

it_t_min = np.min(needy_data.Item_Type)
it_t_max = np.max(needy_data.Item_Type)
#it_wt_max = np.max(needy_data.Item_Weight)
#it_wt_min = np.min(needy_data.Item_Weight)
it_p_max = np.max(needy_data.Item_MRP)
it_p_min = np.min(needy_data.Item_MRP)
it_s_min = np.min(target.Item_Outlet_Sales)
it_s_max = np.max(target.Item_Outlet_Sales)

for l in range(len(needy_data)):
    needy_data.Item_Type[l] = ((needy_data.Item_Type[l]-it_t_min)/(it_t_max - it_t_min))
    #needy_data.Item_Weight[l] = ((needy_data.Item_Weight[l] - it_wt_min)/(it_wt_max - it_wt_min))
    needy_data.Item_MRP[l] = ((needy_data.Item_MRP[l] - it_p_min)/(it_p_max - it_p_min))
    target.Item_Outlet_Sales[l] = ((target.Item_Outlet_Sales[l] - it_s_min)/(it_s_max - it_s_min))


gbm = GradientBoostingRegressor(loss = 'huber', n_estimators = 60, learning_rate = 0.05, random_state = 0)

model = gbm.fit(needy_data, target)


score = cross_val_score(model, needy_data, target['Item_Outlet_Sales'].copy(), cv = 10)

def mean_scores(scores):
    return("Mean score: {0:.3f}(+/-{1:.3f})").format(np.mean(scores), sem(scores))
    
print(mean_scores(score))
    
t_data = test[(test.Outlet_Identifier == 18)].reset_index(drop = True)

test_data = pd.DataFrame({'Item_Type':t_data.Item_Type, 'Item_MRP':t_data.Item_MRP})


ordered_type = []
for q in range(len(test_data.Item_Type.unique())):
    ordered_type.append(test_data.Item_Type.unique()[q])

test_data.Item_Type = test_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
test_data.Item_Type = test_data.Item_Type.astype(float)

for r in range(len(test_data)):
    test_data.Item_Type[r] = ((test_data.Item_Type[r]-it_t_min)/(it_t_max - it_t_min))
    test_data.Item_MRP[r] = ((test_data.Item_MRP[r] - it_p_min)/(it_p_max - it_p_min))
    #test_data.Item_Weight[r] = ((test_data.Item_Weight[r] - it_wt_min)/(it_wt_max - it_wt_min))
    
pred = []
predictions = gbm.predict(test_data)
for z in range(len(predictions)):
    pred.append(predictions[z])
        
for y in range(len(pred)):
    pred[y] = (pred[y]*(it_s_max-it_s_min)) + (it_s_min)
        
for x in range(len(pred)):
    prediction_list.append(pred[x])
    
    
data = sale_data[(sale_data.Outlet_Identifier == 19)].reset_index(drop = True)
needy_data = pd.DataFrame({'Item_Type':data.Item_Type, 'Item_MRP':data.Item_MRP})
target = pd.DataFrame({'Item_Outlet_Sales':data.Item_Outlet_Sales})
    
needy_data = needy_data.fillna(method = 'bfill', axis = 0)
needy_data = needy_data.fillna(method = 'ffill', axis = 0)
needy_data = needy_data.fillna(0)
   
target = target.fillna(method = 'bfill', axis = 0)
target = target.fillna(method = 'ffill', axis = 0)
target = target.fillna(0)

ordered_type = []
for k in range(len(needy_data.Item_Type.unique())):
    ordered_type.append(needy_data.Item_Type.unique()[k])

needy_data.Item_Type = needy_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
needy_data.Item_Type = needy_data.Item_Type.astype(float)

it_t_min = np.min(needy_data.Item_Type)
it_t_max = np.max(needy_data.Item_Type)
#it_wt_max = np.max(needy_data.Item_Weight)
#it_wt_min = np.min(needy_data.Item_Weight)
it_p_max = np.max(needy_data.Item_MRP)
it_p_min = np.min(needy_data.Item_MRP)
it_s_min = np.min(target.Item_Outlet_Sales)
it_s_max = np.max(target.Item_Outlet_Sales)

for l in range(len(needy_data)):
    needy_data.Item_Type[l] = ((needy_data.Item_Type[l]-it_t_min)/(it_t_max - it_t_min))
    #needy_data.Item_Weight[l] = ((needy_data.Item_Weight[l] - it_wt_min)/(it_wt_max - it_wt_min))
    needy_data.Item_MRP[l] = ((needy_data.Item_MRP[l] - it_p_min)/(it_p_max - it_p_min))
    target.Item_Outlet_Sales[l] = ((target.Item_Outlet_Sales[l] - it_s_min)/(it_s_max - it_s_min))


gbm = GradientBoostingRegressor(loss = 'huber', n_estimators = 50, learning_rate = 0.05, random_state = 0)

model = gbm.fit(needy_data, target)


score = cross_val_score(model, needy_data, target['Item_Outlet_Sales'].copy(), cv = 10)

def mean_scores(scores):
    return("Mean score: {0:.3f}(+/-{1:.3f})").format(np.mean(scores), sem(scores))
    
print(mean_scores(score))
    
t_data = test[(test.Outlet_Identifier == 19)].reset_index(drop = True)

test_data = pd.DataFrame({'Item_Type':t_data.Item_Type, 'Item_MRP':t_data.Item_MRP})


ordered_type = []
for q in range(len(test_data.Item_Type.unique())):
    ordered_type.append(test_data.Item_Type.unique()[q])

test_data.Item_Type = test_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
test_data.Item_Type = test_data.Item_Type.astype(float)

for r in range(len(test_data)):
    test_data.Item_Type[r] = ((test_data.Item_Type[r]-it_t_min)/(it_t_max - it_t_min))
    test_data.Item_MRP[r] = ((test_data.Item_MRP[r] - it_p_min)/(it_p_max - it_p_min))
    #test_data.Item_Weight[r] = ((test_data.Item_Weight[r] - it_wt_min)/(it_wt_max - it_wt_min))
    
pred = []
predictions = gbm.predict(test_data)
for z in range(len(predictions)):
    pred.append(predictions[z])
        
for y in range(len(pred)):
    pred[y] = (pred[y]*(it_s_max-it_s_min)) + (it_s_min)
        
for x in range(len(pred)):
    prediction_list.append(pred[x])
    
    
data = sale_data[(sale_data.Outlet_Identifier == 27)].reset_index(drop = True)
needy_data = pd.DataFrame({'Item_Type':data.Item_Type, 'Item_MRP':data.Item_MRP})
target = pd.DataFrame({'Item_Outlet_Sales':data.Item_Outlet_Sales})
    
needy_data = needy_data.fillna(method = 'bfill', axis = 0)
needy_data = needy_data.fillna(method = 'ffill', axis = 0)
needy_data = needy_data.fillna(0)
   
target = target.fillna(method = 'bfill', axis = 0)
target = target.fillna(method = 'ffill', axis = 0)
target = target.fillna(0)

ordered_type = []
for k in range(len(needy_data.Item_Type.unique())):
    ordered_type.append(needy_data.Item_Type.unique()[k])

needy_data.Item_Type = needy_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
needy_data.Item_Type = needy_data.Item_Type.astype(float)

it_t_min = np.min(needy_data.Item_Type)
it_t_max = np.max(needy_data.Item_Type)
#it_wt_max = np.max(needy_data.Item_Weight)
#it_wt_min = np.min(needy_data.Item_Weight)
it_p_max = np.max(needy_data.Item_MRP)
it_p_min = np.min(needy_data.Item_MRP)
it_s_min = np.min(target.Item_Outlet_Sales)
it_s_max = np.max(target.Item_Outlet_Sales)

for l in range(len(needy_data)):
    needy_data.Item_Type[l] = ((needy_data.Item_Type[l]-it_t_min)/(it_t_max - it_t_min))
    #needy_data.Item_Weight[l] = ((needy_data.Item_Weight[l] - it_wt_min)/(it_wt_max - it_wt_min))
    needy_data.Item_MRP[l] = ((needy_data.Item_MRP[l] - it_p_min)/(it_p_max - it_p_min))
    target.Item_Outlet_Sales[l] = ((target.Item_Outlet_Sales[l] - it_s_min)/(it_s_max - it_s_min))


gbm = GradientBoostingRegressor(loss = 'lad', n_estimators = 70, learning_rate = 0.05, random_state = 0)

model = gbm.fit(needy_data, target)


score = cross_val_score(model, needy_data, target['Item_Outlet_Sales'].copy(), cv = 10)

def mean_scores(scores):
    return("Mean score: {0:.3f}(+/-{1:.3f})").format(np.mean(scores), sem(scores))
    
print(mean_scores(score))
    
t_data = test[(test.Outlet_Identifier == 27)].reset_index(drop = True)

test_data = pd.DataFrame({'Item_Type':t_data.Item_Type, 'Item_MRP':t_data.Item_MRP})


ordered_type = []
for q in range(len(test_data.Item_Type.unique())):
    ordered_type.append(test_data.Item_Type.unique()[q])

test_data.Item_Type = test_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
test_data.Item_Type = test_data.Item_Type.astype(float)

for r in range(len(test_data)):
    test_data.Item_Type[r] = ((test_data.Item_Type[r]-it_t_min)/(it_t_max - it_t_min))
    test_data.Item_MRP[r] = ((test_data.Item_MRP[r] - it_p_min)/(it_p_max - it_p_min))
    #test_data.Item_Weight[r] = ((test_data.Item_Weight[r] - it_wt_min)/(it_wt_max - it_wt_min))
    
pred = []
predictions = gbm.predict(test_data)
for z in range(len(predictions)):
    pred.append(predictions[z])
        
for y in range(len(pred)):
    pred[y] = (pred[y]*(it_s_max-it_s_min)) + (it_s_min)
        
for x in range(len(pred)):
    prediction_list.append(pred[x])
    
    
data = sale_data[(sale_data.Outlet_Identifier == 35)].reset_index(drop = True)
needy_data = pd.DataFrame({'Item_Type':data.Item_Type, 'Item_MRP':data.Item_MRP})
target = pd.DataFrame({'Item_Outlet_Sales':data.Item_Outlet_Sales})
    
needy_data = needy_data.fillna(method = 'bfill', axis = 0)
needy_data = needy_data.fillna(method = 'ffill', axis = 0)
needy_data = needy_data.fillna(0)
   
target = target.fillna(method = 'bfill', axis = 0)
target = target.fillna(method = 'ffill', axis = 0)
target = target.fillna(0)

ordered_type = []
for k in range(len(needy_data.Item_Type.unique())):
    ordered_type.append(needy_data.Item_Type.unique()[k])

needy_data.Item_Type = needy_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
needy_data.Item_Type = needy_data.Item_Type.astype(float)

it_t_min = np.min(needy_data.Item_Type)
it_t_max = np.max(needy_data.Item_Type)
#it_wt_max = np.max(needy_data.Item_Weight)
#it_wt_min = np.min(needy_data.Item_Weight)
it_p_max = np.max(needy_data.Item_MRP)
it_p_min = np.min(needy_data.Item_MRP)
it_s_min = np.min(target.Item_Outlet_Sales)
it_s_max = np.max(target.Item_Outlet_Sales)

for l in range(len(needy_data)):
    needy_data.Item_Type[l] = ((needy_data.Item_Type[l]-it_t_min)/(it_t_max - it_t_min))
    #needy_data.Item_Weight[l] = ((needy_data.Item_Weight[l] - it_wt_min)/(it_wt_max - it_wt_min))
    needy_data.Item_MRP[l] = ((needy_data.Item_MRP[l] - it_p_min)/(it_p_max - it_p_min))
    target.Item_Outlet_Sales[l] = ((target.Item_Outlet_Sales[l] - it_s_min)/(it_s_max - it_s_min))


gbm = GradientBoostingRegressor(loss = 'lad', n_estimators = 60, learning_rate = 0.05, random_state = 0)

model = gbm.fit(needy_data, target)


score = cross_val_score(model, needy_data, target['Item_Outlet_Sales'].copy(), cv = 10)

def mean_scores(scores):
    return("Mean score: {0:.3f}(+/-{1:.3f})").format(np.mean(scores), sem(scores))
    
print(mean_scores(score))
    
t_data = test[(test.Outlet_Identifier == 35)].reset_index(drop = True)

test_data = pd.DataFrame({'Item_Type':t_data.Item_Type, 'Item_MRP':t_data.Item_MRP})


ordered_type = []
for q in range(len(test_data.Item_Type.unique())):
    ordered_type.append(test_data.Item_Type.unique()[q])

test_data.Item_Type = test_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
test_data.Item_Type = test_data.Item_Type.astype(float)

for r in range(len(test_data)):
    test_data.Item_Type[r] = ((test_data.Item_Type[r]-it_t_min)/(it_t_max - it_t_min))
    test_data.Item_MRP[r] = ((test_data.Item_MRP[r] - it_p_min)/(it_p_max - it_p_min))
    #test_data.Item_Weight[r] = ((test_data.Item_Weight[r] - it_wt_min)/(it_wt_max - it_wt_min))
    
pred = []
predictions = gbm.predict(test_data)
for z in range(len(predictions)):
    pred.append(predictions[z])
        
for y in range(len(pred)):
    pred[y] = (pred[y]*(it_s_max-it_s_min)) + (it_s_min)
        
for x in range(len(pred)):
    prediction_list.append(pred[x])
    
    
data = sale_data[(sale_data.Outlet_Identifier == 45)].reset_index(drop = True)
needy_data = pd.DataFrame({'Item_Weight':data.Item_Weight, 'Item_Type':data.Item_Type, 'Item_MRP':data.Item_MRP})
target = pd.DataFrame({'Item_Outlet_Sales':data.Item_Outlet_Sales})
    
needy_data = needy_data.fillna(method = 'bfill', axis = 0)
needy_data = needy_data.fillna(method = 'ffill', axis = 0)
needy_data = needy_data.fillna(0)
   
target = target.fillna(method = 'bfill', axis = 0)
target = target.fillna(method = 'ffill', axis = 0)
target = target.fillna(0)

ordered_type = []
for k in range(len(needy_data.Item_Type.unique())):
    ordered_type.append(needy_data.Item_Type.unique()[k])

needy_data.Item_Type = needy_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
needy_data.Item_Type = needy_data.Item_Type.astype(float)

it_t_min = np.min(needy_data.Item_Type)
it_t_max = np.max(needy_data.Item_Type)
it_wt_max = np.max(needy_data.Item_Weight)
it_wt_min = np.min(needy_data.Item_Weight)
it_p_max = np.max(needy_data.Item_MRP)
it_p_min = np.min(needy_data.Item_MRP)
it_s_min = np.min(target.Item_Outlet_Sales)
it_s_max = np.max(target.Item_Outlet_Sales)

for l in range(len(needy_data)):
    needy_data.Item_Type[l] = ((needy_data.Item_Type[l]-it_t_min)/(it_t_max - it_t_min))
    needy_data.Item_Weight[l] = ((needy_data.Item_Weight[l] - it_wt_min)/(it_wt_max - it_wt_min))
    needy_data.Item_MRP[l] = ((needy_data.Item_MRP[l] - it_p_min)/(it_p_max - it_p_min))
    target.Item_Outlet_Sales[l] = ((target.Item_Outlet_Sales[l] - it_s_min)/(it_s_max - it_s_min))


gbm = GradientBoostingRegressor(loss = 'huber', n_estimators = 50, learning_rate = 0.05, random_state = 0)

model = gbm.fit(needy_data, target)


score = cross_val_score(model, needy_data, target['Item_Outlet_Sales'].copy(), cv = 10)

def mean_scores(scores):
    return("Mean score: {0:.3f}(+/-{1:.3f})").format(np.mean(scores), sem(scores))
    
print(mean_scores(score))
    
t_data = test[(test.Outlet_Identifier == 45)].reset_index(drop = True)

test_data = pd.DataFrame({'Item_Weight':t_data.Item_Weight, 'Item_Type':t_data.Item_Type, 'Item_MRP':t_data.Item_MRP})


ordered_type = []
for q in range(len(test_data.Item_Type.unique())):
    ordered_type.append(test_data.Item_Type.unique()[q])

test_data.Item_Type = test_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
test_data.Item_Type = test_data.Item_Type.astype(float)

for r in range(len(test_data)):
    test_data.Item_Type[r] = ((test_data.Item_Type[r]-it_t_min)/(it_t_max - it_t_min))
    test_data.Item_MRP[r] = ((test_data.Item_MRP[r] - it_p_min)/(it_p_max - it_p_min))
    test_data.Item_Weight[r] = ((test_data.Item_Weight[r] - it_wt_min)/(it_wt_max - it_wt_min))
    
pred = []
predictions = gbm.predict(test_data)
for z in range(len(predictions)):
    pred.append(predictions[z])
        
for y in range(len(pred)):
    pred[y] = (pred[y]*(it_s_max-it_s_min)) + (it_s_min)
        
for x in range(len(pred)):
    prediction_list.append(pred[x])
    
    
data = sale_data[(sale_data.Outlet_Identifier == 46)].reset_index(drop = True)
needy_data = pd.DataFrame({'Item_Type':data.Item_Type, 'Item_MRP':data.Item_MRP})
target = pd.DataFrame({'Item_Outlet_Sales':data.Item_Outlet_Sales})
    
needy_data = needy_data.fillna(method = 'bfill', axis = 0)
needy_data = needy_data.fillna(method = 'ffill', axis = 0)
needy_data = needy_data.fillna(0)
   
target = target.fillna(method = 'bfill', axis = 0)
target = target.fillna(method = 'ffill', axis = 0)
target = target.fillna(0)

ordered_type = []
for k in range(len(needy_data.Item_Type.unique())):
    ordered_type.append(needy_data.Item_Type.unique()[k])

needy_data.Item_Type = needy_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
needy_data.Item_Type = needy_data.Item_Type.astype(float)

it_t_min = np.min(needy_data.Item_Type)
it_t_max = np.max(needy_data.Item_Type)
#it_wt_max = np.max(needy_data.Item_Weight)
#it_wt_min = np.min(needy_data.Item_Weight)
it_p_max = np.max(needy_data.Item_MRP)
it_p_min = np.min(needy_data.Item_MRP)
it_s_min = np.min(target.Item_Outlet_Sales)
it_s_max = np.max(target.Item_Outlet_Sales)

for l in range(len(needy_data)):
    needy_data.Item_Type[l] = ((needy_data.Item_Type[l]-it_t_min)/(it_t_max - it_t_min))
    #needy_data.Item_Weight[l] = ((needy_data.Item_Weight[l] - it_wt_min)/(it_wt_max - it_wt_min))
    needy_data.Item_MRP[l] = ((needy_data.Item_MRP[l] - it_p_min)/(it_p_max - it_p_min))
    target.Item_Outlet_Sales[l] = ((target.Item_Outlet_Sales[l] - it_s_min)/(it_s_max - it_s_min))


gbm = GradientBoostingRegressor(loss = 'ls', n_estimators = 50, learning_rate = 0.05, random_state = 0)

model = gbm.fit(needy_data, target)


score = cross_val_score(model, needy_data, target['Item_Outlet_Sales'].copy(), cv = 10)

def mean_scores(scores):
    return("Mean score: {0:.3f}(+/-{1:.3f})").format(np.mean(scores), sem(scores))
    
print(mean_scores(score))
    
t_data = test[(test.Outlet_Identifier == 46)].reset_index(drop = True)

test_data = pd.DataFrame({'Item_Type':t_data.Item_Type, 'Item_MRP':t_data.Item_MRP})


ordered_type = []
for q in range(len(test_data.Item_Type.unique())):
    ordered_type.append(test_data.Item_Type.unique()[q])

test_data.Item_Type = test_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
test_data.Item_Type = test_data.Item_Type.astype(float)

for r in range(len(test_data)):
    test_data.Item_Type[r] = ((test_data.Item_Type[r]-it_t_min)/(it_t_max - it_t_min))
    test_data.Item_MRP[r] = ((test_data.Item_MRP[r] - it_p_min)/(it_p_max - it_p_min))
    #test_data.Item_Weight[r] = ((test_data.Item_Weight[r] - it_wt_min)/(it_wt_max - it_wt_min))
    
pred = []
predictions = gbm.predict(test_data)
for z in range(len(predictions)):
    pred.append(predictions[z])
        
for y in range(len(pred)):
    pred[y] = (pred[y]*(it_s_max-it_s_min)) + (it_s_min)
        
for x in range(len(pred)):
    prediction_list.append(pred[x])
    
    
data = sale_data[(sale_data.Outlet_Identifier == 49)].reset_index(drop = True)
needy_data = pd.DataFrame({'Item_Type':data.Item_Type, 'Item_MRP':data.Item_MRP})
target = pd.DataFrame({'Item_Outlet_Sales':data.Item_Outlet_Sales})
    
needy_data = needy_data.fillna(method = 'bfill', axis = 0)
needy_data = needy_data.fillna(method = 'ffill', axis = 0)
needy_data = needy_data.fillna(0)
   
target = target.fillna(method = 'bfill', axis = 0)
target = target.fillna(method = 'ffill', axis = 0)
target = target.fillna(0)

ordered_type = []
for k in range(len(needy_data.Item_Type.unique())):
    ordered_type.append(needy_data.Item_Type.unique()[k])

needy_data.Item_Type = needy_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
needy_data.Item_Type = needy_data.Item_Type.astype(float)

it_t_min = np.min(needy_data.Item_Type)
it_t_max = np.max(needy_data.Item_Type)
#it_wt_max = np.max(needy_data.Item_Weight)
#it_wt_min = np.min(needy_data.Item_Weight)
it_p_max = np.max(needy_data.Item_MRP)
it_p_min = np.min(needy_data.Item_MRP)
it_s_min = np.min(target.Item_Outlet_Sales)
it_s_max = np.max(target.Item_Outlet_Sales)

for l in range(len(needy_data)):
    needy_data.Item_Type[l] = ((needy_data.Item_Type[l]-it_t_min)/(it_t_max - it_t_min))
    #needy_data.Item_Weight[l] = ((needy_data.Item_Weight[l] - it_wt_min)/(it_wt_max - it_wt_min))
    needy_data.Item_MRP[l] = ((needy_data.Item_MRP[l] - it_p_min)/(it_p_max - it_p_min))
    target.Item_Outlet_Sales[l] = ((target.Item_Outlet_Sales[l] - it_s_min)/(it_s_max - it_s_min))


gbm = GradientBoostingRegressor(loss = 'ls', n_estimators = 50, learning_rate = 0.05, random_state = 0)

model = gbm.fit(needy_data, target)


score = cross_val_score(model, needy_data, target['Item_Outlet_Sales'].copy(), cv = 10)

def mean_scores(scores):
    return("Mean score: {0:.3f}(+/-{1:.3f})").format(np.mean(scores), sem(scores))
    
print(mean_scores(score))
    
t_data = test[(test.Outlet_Identifier == 49)].reset_index(drop = True)

test_data = pd.DataFrame({'Item_Type':t_data.Item_Type, 'Item_MRP':t_data.Item_MRP})


ordered_type = []
for q in range(len(test_data.Item_Type.unique())):
    ordered_type.append(test_data.Item_Type.unique()[q])

test_data.Item_Type = test_data.Item_Type.astype("category", ordered=True, categories=ordered_type).cat.codes
test_data.Item_Type = test_data.Item_Type.astype(float)

for r in range(len(test_data)):
    test_data.Item_Type[r] = ((test_data.Item_Type[r]-it_t_min)/(it_t_max - it_t_min))
    test_data.Item_MRP[r] = ((test_data.Item_MRP[r] - it_p_min)/(it_p_max - it_p_min))
    #test_data.Item_Weight[r] = ((test_data.Item_Weight[r] - it_wt_min)/(it_wt_max - it_wt_min))
    
pred = []
predictions = gbm.predict(test_data)
for z in range(len(predictions)):
    pred.append(predictions[z])
        
for y in range(len(pred)):
    pred[y] = (pred[y]*(it_s_max-it_s_min)) + (it_s_min)
        
for x in range(len(pred)):
    prediction_list.append(pred[x])
        
submission = pd.read_csv('/home/Akai/Downloads/Big Mart III/sample.csv')

submission['Item_Outlet_Sales'] = pd.DataFrame({'Item_Outlet_Sales':prediction_list})
submission.to_csv('submission.csv')