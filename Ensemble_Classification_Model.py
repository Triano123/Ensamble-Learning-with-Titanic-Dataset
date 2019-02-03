#module analisys
import pandas as pd 
import numpy as np 

#module machine learning
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, auc

#save model 
from sklearn.externals import joblib

#module for imbalance-learn for balancing class
from imblearn.over_sampling import SMOTE

#visualization
import matplotlib.pyplot as plt 
import seaborn as sns 

# Module Ensamble Method
def ensemble_classification_model(df, target ,imbalance = False, algorithm = None) :
	'''
	Parameters :
	------------

	* df         :     object, dataframe
			       The dataframe that have been passed pre processing steps. 
			       ex : (EDA,feature selection, feature engineering).
	* target     :     string
				   The target is the feature of a dataset about which you want 
				   to gain a deeper understanding.
	* imbalance  : 	   boolean, default False
				   > if imbalance is False, the model will use train and test split usually.
				   > if imbalance is True, the model will use balancing methode for handling 
				     the imbalance the class.
				   Imbalanced data sets are a special case for classification problem where 
				   the class distribution is not uniform among the classes.
	* algorithm  :     integer
				   > [0] : Decision Tree Algorithm 
				   > [1] : Random Forest Algorithm
				   > [2] : Boostrap Aggregating Algorithm (Bagging)
				   > [3] : Adaptive Boosting Algorithm (Adsboost)
				   > [4] : Gradient Boosting  Algorithm
	* save_model :    Boolean, default True
				   > if True  = Save Model
				   > if False = View ROC Curve and Accuracy test

	Example    :
	------------
	> Input dataframe/dataset :
	   df = df_home_price
	> define target :
	   target = 'Sales' 
	> Data Imbalance or not 
	   imbalance = True
	> The algorithm that will used 
	   algorithm = 1 #Random Forest Algorithm
	> Module 
	    ensemble_classification_model(df = df_home_price,target = 'Sales',imbalance = True,algorithm = 1) 
	'''
	#train test split :
	if imbalance is False :
		X = df.drop([target],axis = 1) 
		y = df[target]
		X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

		treshold = 15
	
		if algorithm is 0 :
			print('The algorithm used is DecisionTree Classifier')
			
			dt = DecisionTreeClassifier(max_depth=10, min_samples_split=3, min_samples_leaf=2)
			dt.fit(X_train, y_train)
			dt_pred_train = dt.predict(X_train)
			dt_pred_test = dt.predict(X_test)

			#Accuracy train
			dt_acc_train = round(accuracy_score(y_train,dt_pred_train)*100,1)
			print('Accuracy train =', dt_acc_train,'%')
			#Accuracy test
			dt_acc_test = round(accuracy_score(y_test,dt_pred_test)*100,1)
			print ('Accuracy Test =', dt_acc_test,'%')

			if round(dt_acc_train - dt_acc_test) > treshold :
				print ('There is indicate Overfitting in the model')

				#need Tuning parameters 
				param = {'splitter': ['best','random'],
				'max_features': [10,30,50],
				'max_depth': [10, 30, 50],
				'min_samples_split': [2, 5, 10],
				'min_samples_leaf': [10, 20, 30],
				'bootstrap': [True, False],
				'criterion': ['gini']}

				#model tuning
				tuning = RandomizedSearchCV(dt, param, n_iter=3, cv= 5)

				# fit randomize search model
				best_model_dt = tuning.fit(X_train,y_train)
				best_pred_train = best_model_dt.predict(X_train)
				best_pred_test  = best_model_dt.predict(X_test)

				#Accuracy train
				best_acc_train = round(accuracy_score(y_train,best_pred_train)*100,1)
				print ('Accuracy best train =', best_pred_train,'%')
				#Accuracy test
				best_acc_test = round(accuracy_score(y_test,best_pred_test)*100,1)
				print ('Accuracy best test =', best_acc_test,'%')
				

				#Roc Curve
				probs = best_model_dt.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =', auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')

			elif round(dt_acc_train - dt_acc_test) < treshold :
				
				print('The model is good')
				#Roc Curve
				probs = dt.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =',  auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')


		if algorithm is 1 :
			print('The algorithm used is Random Forest Classifier')
			
			rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=5)
			rf.fit(X_train, y_train)
			rf_pred_train = rf.predict(X_train)
			rf_pred_test = rf.predict(X_test)

			#Accuracy train
			rf_acc_train = round(accuracy_score(y_train,rf_pred_train)*100,1)
			print('Accuracy train =', rf_acc_train,'%')
			#Accuracy test
			rf_acc_test = round(accuracy_score(y_test,rf_pred_test)*100,1)
			print ('Accuracy Test =', rf_acc_test,'%')

			if round(rf_acc_train - rf_acc_test) > treshold :
				print ('There is indicate Overfitting in the model')

				#need Tuning parameters 
				param = {'n_estimators': [100, 300, 500],
				'max_features': [10,30,50],
				'max_depth': [10, 30, 50],
				'min_samples_split': [2, 5, 10],
				'min_samples_leaf': [10, 20, 30],
				'bootstrap': [True, False],
				'criterion': ['gini']}

				#model tuning
				tuning = RandomizedSearchCV(rf, param, n_iter=3, cv= 5)

				# fit randomize search model
				best_model_rf = tuning.fit(X_train,y_train)
				best_pred_train = best_model_rf.predict(X_train)
				best_pred_test  = best_model_rf.predict(X_test)

				#Accuracy train
				best_acc_train = round(accuracy_score(y_train,best_pred_train)*100,1)
				print ('Accuracy best train =', best_pred_train,'%')
				#Accuracy test
				best_acc_test = round(accuracy_score(y_test,best_pred_test)*100,1)
				print ('Accuracy best test =', best_acc_test,'%')
				

				#Roc Curve
				probs = best_model_rf.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =', auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')

			elif round(rf_acc_train - rf_acc_test) < treshold :
				
				print('The model is good')
				#Roc Curve
				probs = rf.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =',  auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')


		if algorithm is 2 :
			print('The algorithm used is Bagging Classifier')
			
			dt = DecisionTreeClassifier()
			bg = BaggingClassifier(base_estimator=dt, n_estimators=200)
			bg.fit(X_train, y_train)
			bg_pred_train = bg.predict(X_train)
			bg_pred_test = bg.predict(X_test)

			#Accuracy train
			bg_acc_train = round(accuracy_score(y_train,bg_pred_train)*100,1)
			print('Accuracy train =', bg_acc_train,'%')
			#Accuracy test
			bg_acc_test = round(accuracy_score(y_test,bg_pred_test)*100,1)
			print ('Accuracy Test =', bg_acc_test,'%')

			if round(bg_acc_train - bg_acc_test) > treshold :
				print ('There is indicate Overfitting in the model')

				#need Tuning parameters 
				param = {'n_estimators': [100, 300, 500],
				'max_features': [10,50],
				'max_samples' :[10,30],
				'bootstrap': [True, False]}

				#model tuning
				tuning = RandomizedSearchCV(bg, param, n_iter=3, cv= 5)

				# fit randomize search model
				best_model_bg = tuning.fit(X_train,y_train)
				best_pred_train = best_model_bg.predict(X_train)
				best_pred_test  = best_model_bg.predict(X_test)

				#Accuracy train
				best_acc_train = round(accuracy_score(y_train,best_pred_train)*100,1)
				print ('Accuracy best train =', best_pred_train,'%')
				#Accuracy test
				best_acc_test = round(accuracy_score(y_test,best_pred_test)*100,1)
				print ('Accuracy best test =', best_acc_test,'%')
				

				#Roc Curve
				probs = best_model_bg.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =', auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')

			elif round(bg_acc_train - bg_acc_test) < treshold :
				
				print('The model is good')
				#Roc Curve
				probs = bg.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =',  auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')


		if algorithm is 3 :
			print('The algorithm used is Adaboost Classifier')
			
			dt = DecisionTreeClassifier()
			ab = AdaBoostClassifier(base_estimator=dt, n_estimators=200)
			ab.fit(X_train, y_train)
			ab_pred_train = ab.predict(X_train)
			ab_pred_test = ab.predict(X_test)

			#Accuracy train
			ab_acc_train = round(accuracy_score(y_train,ab_pred_train)*100,1)
			print('Accuracy train =', ab_acc_train,'%')
			#Accuracy test
			ab_acc_test = round(accuracy_score(y_test,ab_pred_test)*100,1)
			print ('Accuracy Test =', ab_acc_test,'%')

			if round(ab_acc_train - ab_acc_test) > treshold :
				print ('There is indicate Overfitting in the model')

				#need Tuning parameters 
				param = {'n_estimators': [100, 300, 500],'learning_rate': [10,50]}

				#model tuning
				tuning = RandomizedSearchCV(ab, param, n_iter=3, cv= 5)

				# fit randomize search model
				best_model_ab = tuning.fit(X_train,y_train)
				best_pred_train = best_model_ab.predict(X_train)
				best_pred_test  = best_model_ab.predict(X_test)

				#Accuracy train
				best_acc_train = round(accuracy_score(y_train,best_pred_train)*100,1)
				print ('Accuracy best train =', best_pred_train,'%')
				#Accuracy test
				best_acc_test = round(accuracy_score(y_test,best_pred_test)*100,1)
				print ('Accuracy best test =', best_acc_test,'%')
				

				#Roc Curve
				probs = best_model_ab.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =', auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')

			elif round(ab_acc_train - ab_acc_test) < treshold :
				
				print('The model is good')
				#Roc Curve
				probs = ab.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =',  auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')


		if algorithm is 4 :
			print('The algorithm used is Gradient Boosting Classifier')
			
			gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.2)
			gbc.fit(X_train, y_train)
			gbc_pred_train = gbc.predict(X_train)
			gbc_pred_test = gbc.predict(X_test)

			#Accuracy train
			gbc_acc_train = round(accuracy_score(y_train,gbc_pred_train)*100,1)
			print('Accuracy train =', gbc_acc_train,'%')
			#Accuracy test
			gbc_acc_test = round(accuracy_score(y_test,gbc_pred_test)*100,1)
			print ('Accuracy Test =', gbc_acc_test,'%')

			if round(gbc_acc_train - gbc_acc_test) > treshold :
				print ('There is indicate Overfitting in the model')

				#need Tuning parameters 
				param = {'n_estimators': [100, 300, 500],'learning_rate': [0.1,0.2,0.5],'loss':['deviance', 'exponential']}

				#model tuning
				tuning = RandomizedSearchCV(gbc, param, n_iter=3, cv= 5)

				# fit randomize search model
				best_model_gbc = tuning.fit(X_train,y_train)
				best_pred_train = best_model_gbc.predict(X_train)
				best_pred_test  = best_model_gbc.predict(X_test)

				#Accuracy train
				best_acc_train = round(accuracy_score(y_train,best_pred_train)*100,1)
				print ('Accuracy best train =', best_pred_train,'%')
				#Accuracy test
				best_acc_test = round(accuracy_score(y_test,best_pred_test)*100,1)
				print ('Accuracy best test =', best_acc_test,'%')
				

				#Roc Curve
				probs = best_model_gbc.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =', auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')

			elif round(gbc_acc_train - gbc_acc_test) < treshold :
				
				print('The model is good')
				#Roc Curve
				probs = gbc.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =',  auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')

	elif imbalance is True :
		X = df.drop(['Survived'],axis = 1) 
		y = df['Survived']
		X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
		# USING SMOTE
		sm = SMOTE(random_state=0)
		X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

		treshold = 15
	
		if algorithm is 0 :
			print('The algorithm used is DecisionTree Classifier')
			
			dt = DecisionTreeClassifier(max_depth=10, min_samples_split=3, min_samples_leaf=2)
			dt.fit(X_train_res, y_train_res)
			dt_pred_train = dt.predict(X_train_res)
			dt_pred_test = dt.predict(X_test)

			#Accuracy train
			dt_acc_train = round(accuracy_score(y_train_res,dt_pred_train)*100,1)
			print('Accuracy train =', dt_acc_train,'%')
			#Accuracy test
			dt_acc_test = round(accuracy_score(y_test,dt_pred_test)*100,1)
			print ('Accuracy Test =', dt_acc_test,'%')

			if round(dt_acc_train - dt_acc_test) > treshold :
				print ('There is indicate Overfitting in the model')

				#need Tuning parameters 
				param = {'splitter': ['best','random'],
				'max_features': [10,30,50],
				'max_depth': [10, 30, 50],
				'min_samples_split': [2, 5, 10],
				'min_samples_leaf': [10, 20, 30],
				'bootstrap': [True, False],
				'criterion': ['gini']}

				#model tuning
				tuning = RandomizedSearchCV(dt, param, n_iter=3, cv= 5)

				# fit randomize search model
				best_model_dt = tuning.fit(X_train_res,y_train_res)
				best_pred_train = best_model_dt.predict(X_train_res)
				best_pred_test  = best_model_dt.predict(X_test)

				#Accuracy train
				best_acc_train = round(accuracy_score(y_train_res,best_pred_train)*100,1)
				print ('Accuracy best train =', best_pred_train,'%')
				#Accuracy test
				best_acc_test = round(accuracy_score(y_test,best_pred_test)*100,1)
				print ('Accuracy best test =', best_acc_test,'%')
				

				#Roc Curve
				probs = best_model_dt.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =', auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')

			elif round(dt_acc_train - dt_acc_test) < treshold :
				
				print('The model is good')
				#Roc Curve
				probs = dt.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =',  auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')


		if algorithm is 1 :
			print('The algorithm used is Random Forest Classifier')
			
			rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=5)
			rf.fit(X_train_res, y_train_res)
			rf_pred_train = rf.predict(X_train_res)
			rf_pred_test = rf.predict(X_test)

			#Accuracy train
			rf_acc_train = round(accuracy_score(y_train_res,rf_pred_train)*100,1)
			print('Accuracy train =', rf_acc_train,'%')
			#Accuracy test
			rf_acc_test = round(accuracy_score(y_test,rf_pred_test)*100,1)
			print ('Accuracy Test =', rf_acc_test,'%')

			if round(rf_acc_train - rf_acc_test) > treshold :
				print ('There is indicate Overfitting in the model')

				#need Tuning parameters 
				param = {'n_estimators': [100, 300, 500],
				'max_features': [10,30,50],
				'max_depth': [10, 30, 50],
				'min_samples_split': [2, 5, 10],
				'min_samples_leaf': [10, 20, 30],
				'bootstrap': [True, False],
				'criterion': ['gini']}

				#model tuning
				tuning = RandomizedSearchCV(rf, param, n_iter=3, cv= 5)

				# fit randomize search model
				best_model_rf = tuning.fit(X_train_res,y_train_res)
				best_pred_train = best_model_rf.predict(X_train_res)
				best_pred_test  = best_model_rf.predict(X_test)

				#Accuracy train
				best_acc_train = round(accuracy_score(y_train_res,best_pred_train)*100,1)
				print ('Accuracy best train =', best_pred_train,'%')
				#Accuracy test
				best_acc_test = round(accuracy_score(y_test,best_pred_test)*100,1)
				print ('Accuracy best test =', best_acc_test,'%')
				

				#Roc Curve
				probs = best_model_rf.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =', auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')

			elif round(rf_acc_train - rf_acc_test) < treshold :
				
				print('The model is good')
				#Roc Curve
				probs = rf.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =',  auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')


		if algorithm is 2 :
			print('The algorithm used is Bagging Classifier')
			
			dt = DecisionTreeClassifier()
			bg = BaggingClassifier(base_estimator=dt, n_estimators=200)
			bg.fit(X_train_res, y_train_res)
			bg_pred_train = bg.predict(X_train_res)
			bg_pred_test = bg.predict(X_test)

			#Accuracy train
			bg_acc_train = round(accuracy_score(y_train_res,bg_pred_train)*100,1)
			print('Accuracy train =', bg_acc_train,'%')
			#Accuracy test
			bg_acc_test = round(accuracy_score(y_test,bg_pred_test)*100,1)
			print ('Accuracy Test =', bg_acc_test,'%')

			if round(bg_acc_train - bg_acc_test) > treshold :
				print ('There is indicate Overfitting in the model')

				#need Tuning parameters 
				param = {'n_estimators': [100, 300, 500],
				'max_features': [10,50],
				'max_samples' :[10,30],
				'bootstrap': [True, False]}

				#model tuning
				tuning = RandomizedSearchCV(bg, param, n_iter=3, cv= 5)

				# fit randomize search model
				best_model_bg = tuning.fit(X_train_res,y_train_res)
				best_pred_train = best_model_bg.predict(X_train_res)
				best_pred_test  = best_model_bg.predict(X_test)

				#Accuracy train
				best_acc_train = round(accuracy_score(y_train_res,best_pred_train)*100,1)
				print ('Accuracy best train =', best_pred_train,'%')
				#Accuracy test
				best_acc_test = round(accuracy_score(y_test,best_pred_test)*100,1)
				print ('Accuracy best test =', best_acc_test,'%')
				

				#Roc Curve
				probs = best_model_bg.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =', auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')

			elif round(bg_acc_train - bg_acc_test) < treshold :
				
				print('The model is good')
				#Roc Curve
				probs = bg.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =',  auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')


		if algorithm is 3 :
			print('The algorithm used is Adaboost Classifier')
			
			dt = DecisionTreeClassifier()
			ab = AdaBoostClassifier(base_estimator=dt, n_estimators=200)
			ab.fit(X_train_res, y_train_res)
			ab_pred_train = ab.predict(X_train_res)
			ab_pred_test = ab.predict(X_test_res)

			#Accuracy train
			ab_acc_train = round(accuracy_score(y_train_res,ab_pred_train)*100,1)
			print('Accuracy train =', ab_acc_train,'%')
			#Accuracy test
			ab_acc_test = round(accuracy_score(y_test,ab_pred_test)*100,1)
			print ('Accuracy Test =', ab_acc_test,'%')

			if round(ab_acc_train - ab_acc_test) > treshold :
				print ('There is indicate Overfitting in the model')

				#need Tuning parameters 
				param = {'n_estimators': [100, 300, 500],'learning_rate': [10,50]}

				#model tuning
				tuning = RandomizedSearchCV(ab, param, n_iter=3, cv= 5)

				# fit randomize search model
				best_model_ab = tuning.fit(X_train_res,y_train_res)
				best_pred_train = best_model_ab.predict(X_train_res)
				best_pred_test  = best_model_ab.predict(X_test)

				#Accuracy train
				best_acc_train = round(accuracy_score(y_train_res,best_pred_train)*100,1)
				print ('Accuracy best train =', best_pred_train,'%')
				#Accuracy test
				best_acc_test = round(accuracy_score(y_test,best_pred_test)*100,1)
				print ('Accuracy best test =', best_acc_test,'%')
				

				#Roc Curve
				probs = best_model_ab.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =', auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')

			elif round(ab_acc_train - ab_acc_test) < treshold :
				
				print('The model is good')
				#Roc Curve
				probs = ab.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =',  auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')


		if algorithm is 4 :
			print('The algorithm used is Gradient Boosting Classifier')
			
			gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.2)
			gbc.fit(X_train_res, y_train_res)
			gbc_pred_train = gbc.predict(X_train_res)
			gbc_pred_test = gbc.predict(X_test)

			#Accuracy train
			gbc_acc_train = round(accuracy_score(y_train_res,gbc_pred_train)*100,1)
			print('Accuracy train =', gbc_acc_train,'%')
			#Accuracy test
			gbc_acc_test = round(accuracy_score(y_test,gbc_pred_test)*100,1)
			print ('Accuracy Test =', gbc_acc_test,'%')

			if round(gbc_acc_train - gbc_acc_test) > treshold :
				print ('There is indicate Overfitting in the model')

				#need Tuning parameters 
				param = {'n_estimators': [100, 300, 500],'learning_rate': [0.1,0.2,0.5],'loss':['deviance', 'exponential']}

				#model tuning
				tuning = RandomizedSearchCV(gbc, param, n_iter=3, cv= 5)

				# fit randomize search model
				best_model_gbc = tuning.fit(X_train_res,y_train_res)
				best_pred_train = best_model_gbc.predict(X_train_res)
				best_pred_test  = best_model_gbc.predict(X_test)

				#Accuracy train
				best_acc_train = round(accuracy_score(y_train_res,best_pred_train)*100,1)
				print ('Accuracy best train =', best_pred_train,'%')
				#Accuracy test
				best_acc_test = round(accuracy_score(y_test,best_pred_test)*100,1)
				print ('Accuracy best test =', best_acc_test,'%')
				

				#Roc Curve
				probs = best_model_gbc.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =', auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')

			elif round(gbc_acc_train - gbc_acc_test) < treshold :
				
				print('The model is good')
				#Roc Curve
				probs = gbc.predict_proba(X_test)
				probs = probs[:, 1]
				auc = round(roc_auc_score(y_test, probs),2)
				print('AUC value =',  auc)
				fpr, tpr, thresholds = roc_curve(y_test, probs)

				# plot the roc curve for the model
				plt.figure(figsize=(12,8))
				plt.plot([0, 1], [0, 1], linestyle='--')
				plt.plot(fpr, tpr, color='red',label = 'AUC = %0.2f' % auc)
				plt.xlabel('1-False Positive Rate')
				plt.ylabel('True Positive Rate')
				plt.title('Receiver Operating Characteristic')
				plt.legend(loc = 'lower right')
