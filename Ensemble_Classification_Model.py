#module analisys
import pandas as pd 
import numpy as np 

#module machine learning classification
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier

#evaluate model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import roc_curve, roc_auc_score, auc, confusion_matrix, f1_score

#save model 
from sklearn.externals import joblib

#module for imbalance-learn for balancing class
from imblearn.over_sampling import SMOTE

#visualization
import matplotlib.pyplot as plt 
import seaborn as sns 

#########################################################
#        	    ENSEMBLE MMETHODS FUNCTION	    		#
#########################################################

# Module Ensamble Method
def ensemble_classification_model(df, target ,imbalance = False, algorithm = None, save_model = False, filename =None) :
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
	* algorithm  :     integer, Default Random Forest Algorithm
				   > None : Random Forest Algorithm
				   > [1]  : Boostrap Aggregating Algorithm (Bagging)
				   > [2]  : Adaptive Boosting Algorithm (Adsboost)
				   > [3]  : Gradient Boosting  Algorithm
	* save_model :    Boolean, default False
				   > if True  = Save Model & View ROC Curve and Accuracy test
				   > if False = View ROC Curve and Accuracy test
	* filname    :     string
				    filname what you wanna save as a model name (filename.sav)

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
	#########################################################
	#        			ROC CURVE FUNCTION					#
	#########################################################
	def ROC_CURVE(feature_train, target_train, feature_test,target_test, model):
		#AUC Train
		probs_train = model.predict_proba(feature_train)
		probs_train = probs_train[:, 1]
		auc_train = round(roc_auc_score(target_train, probs_train),2)
		fpr1, tpr1, thresholds1 = roc_curve(target_train, probs_train)
		print('AUC Train =',auc_train)

		#AUC test
		probs_test = model.predict_proba(feature_test)
		probs_test= probs_test[:, 1]
		auc_test = round(roc_auc_score(target_test, probs_test),2)
		fpr2, tpr2, thresholds2 = roc_curve(target_test, probs_test)
		print('AUC Test =',auc_test)

		# plot the roc curve for the model
		plt.figure(figsize=(12,8))
		plt.plot([0, 1], [0, 1], linestyle='--')
		plt.plot(fpr1, tpr1, color='red',label = 'AUC Train = %0.2f' % auc_train)
		plt.plot(fpr2, tpr2, color='orange',label = 'AUC Test = %0.2f' % auc_test)
		plt.xlabel('1-False Positive Rate')
		plt.ylabel('True Positive Rate')
		plt.title('Receiver Operating Characteristic')
		plt.legend(loc = 'lower right')

#########################################################
#        		EVALUATE MODEL FUNCTION					#
#########################################################
	def evaluate_model(train_target,train_predict, test_target, test_predict):
		confusion_matrix_train = confusion_matrix(train_target, train_predict)
		print('Confusion_matrix_train =','\n',confusion_matrix_train)
		print('_'*20,'\n')
		confusion_matrix_test = confusion_matrix(test_target, test_predict)
		print('Confusion_matrix_test =','\n',confusion_matrix_test)
		print('_'*20,'\n')
		train_accuracy = round(accuracy_score(train_target,train_predict)*100,1)
		print('Accuracy Train =','\n',train_accuracy)
		print('_'*20,'\n')
		test_accuracy = round(accuracy_score(test_target,test_predict)*100,1)
		print('Accuracy Test =','\n',test_accuracy)
		print('_'*20)

		return (confusion_matrix_train, confusion_matrix_test, train_accuracy, test_accuracy)


#########################################################
#        		   SAVE MODEL FUNCTION					#
#########################################################
	def save(model, filename):
		filename = filename
		joblib.dump(model, filename)

   	####################################### MACHINE LEARNING ALGORITHM #######################################

    #########################################################
	#				  RANDOM FOREST FUNCTION				#
	#########################################################
	def RandomForest(X_train, X_test, y_train, y_test):
		rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=5)
		rf.fit(X_train, y_train)
		#predict
		rf_pred_train = rf.predict(X_train)
		rf_pred_test = rf.predict(X_test)

		return (rf,rf_pred_train,rf_pred_test)

	#########################################################
	#				     BAGGING FUNCTION		     		#
	#########################################################
	def Bagging(X_train, X_test, y_train, y_test):
		dt = DecisionTreeClassifier()
		bg = BaggingClassifier(base_estimator=dt, n_estimators=200)
		bg.fit(X_train, y_train)
		bg_pred_train = bg.predict(X_train)
		bg_pred_test = bg.predict(X_test)

		return (bg, bg_pred_train, bg_pred_test)


	#########################################################
	#				    ADABOOST FUNCTION	     			#
	#########################################################
	def adaboost(X_train, X_test, y_train, y_test):
		dt = DecisionTreeClassifier()
		ab = AdaBoostClassifier(base_estimator=dt)
		ab.fit(X_train, y_train)
		ab_pred_train = ab.predict(X_train)
		ab_pred_test = ab.predict(X_test)

		return (ab, ab_pred_train,ab_pred_test)

	#########################################################
	#			    GRADIENT BOOSTING FUNCTION	        	#
	#########################################################
	def gradientboosting(X_train, X_test, y_train, y_test):
		gbc = GradientBoostingClassifier(n_estimators=200, learning_rate=0.2)
		gbc.fit(X_train, y_train)
		gbc_pred_train = gbc.predict(X_train)
		gbc_pred_test = gbc.predict(X_test)

		return (gbc, gbc_pred_train,gbc_pred_test)


	#########################################################
	#			   RANDOMIZEDSEARCHCV FUNCTION              #
	#########################################################
	def model_tuning(model, parameter, X_train, y_train):
		#model tuning
		tuning = RandomizedSearchCV(model, parameter, n_iter=3, cv= 5)
		# fit randomize search model
		best_model_tuning = tuning.fit(X_train,y_train)
		best_pred_train = best_model_tuning.predict(X_train)
		best_pred_test  = best_model_tuning.predict(X_test)

		return (best_pred_train,best_pred_test)


####################################################################################################
####################################### IF  BALANCE CLASS ##########################################
	#train test split :
	if imbalance is False :
		X = df.drop([target],axis = 1) 
		y = df[target]
		X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

		treshold_1 = 15
		treshold_2 = 70

################################# Random Forest Classifier #################################
		if algorithm is None :
			print('The algorithm = Random Forest Classifier','\n')
			
			#model
			rf,rf_pred_train, rf_pred_test=RandomForest(X_train, X_test, y_train, y_test)

			#Evaluate Model 
			_,_,train_accuracy, test_accuracy = evaluate_model(y_train, rf_pred_train, y_test, rf_pred_test)

			
			if  round(train_accuracy) < treshold_2 :
				print (' The Model is Underfitting')

			elif round(train_accuracy - test_accuracy) > treshold_1 :
				print ('There is indicate Overfitting in the model')

				#Tuning Parameters 
				param = {'n_estimators': [100, 500],
				'max_features': [10,50],
				'max_depth': [5, 10, 20],
				'min_samples_split': [2, 5, 10],
				'min_samples_leaf': [10, 20, 30],
				'bootstrap': [True, False],
				'criterion': ['gini']}

				best_pred_train,best_pred_test=model_tuning(rf, param, X_train, y_train)

				#Evaluate Model 
				_,_,train_accuracy, test_accuracy = evaluate_model(y_train, best_pred_train, y_test, best_pred_test)

				if save_model is False :
					#Roc Curve
					ROC_CURVE(X_train,y_train,X_test,y_test,rf)
				elif save_model is True :
					filename  = filename
					save(rf,filename)
					#Roc Curve
					ROC_CURVE(X_train,y_train,X_test,y_test,rf)

			elif round(train_accuracy - test_accuracy) < treshold_1:
				
				if save_model is False :
					print('The model is good','\n')
					#Roc Curve
					ROC_CURVE(X_train,y_train,X_test,y_test,rf)
				elif save_model is True :
					print('The model is good','\n')
					filename  = filename
					save(rf,filename)
					#Roc Curve
					ROC_CURVE(X_train,y_train,X_test,y_test,rf)
				
################################# Bagging Classifier #################################

		if algorithm is 1 :
			print('The algorithm = Bagging Classifier (Base Estimator is DecisionTree)','\n')
			
			#model
			bg,bg_pred_train, bg_pred_test=Bagging(X_train, X_test, y_train, y_test)

			#Evaluate Model 
			_,_,train_accuracy, test_accuracy = evaluate_model(y_train, bg_pred_train, y_test, bg_pred_test)

			
			if  round(train_accuracy) < treshold_2 :
				print (' The Model is Underfitting')
			

			elif round(train_accuracy - test_accuracy) > treshold_1 :
				print ('There is indicate Overfitting in the model')

				#need Tuning parameters 
				param = {'n_estimators': [100,300,500],
				'max_features': [10,50],
				'max_samples' :[10,30],
				'bootstrap': [True, False]}

				best_pred_train,best_pred_test=model_tuning(bg, param, X_train, y_train)

				#Evaluate Model 
				_,_,train_accuracy, test_accuracy = evaluate_model(y_train, best_pred_train, y_test, best_pred_test)

				if save_model is False :
					#Roc Curve
					ROC_CURVE(X_train,y_train,X_test,y_test,bg)
				elif save_model is True :
					filename  = filename
					save(bg,filename)
					#Roc Curve
					ROC_CURVE(X_train,y_train,X_test,y_test,bg)

			elif round(train_accuracy - test_accuracy) < treshold_1:
				
				if save_model is False :
					print('The model is good','\n')
					#Roc Curve
					ROC_CURVE(X_train,y_train,X_test,y_test,bg)
				elif save_model is True :
					print('The model is good','\n')
					filename  = filename
					save(bg,filename)
					#Roc Curve
					ROC_CURVE(X_train,y_train,X_test,y_test,bg)

				
################################# Adaboost Classifier #################################

		if algorithm is 2 :
			print('The algorithm = Adaboost Classifier (Base Estimator is DecisionTree)','\n')
			
			#model
			ab,ab_pred_train, ab_pred_test=adaboost(X_train, X_test, y_train, y_test)

			#Evaluate Model 
			_,_,train_accuracy, test_accuracy = evaluate_model(y_train, ab_pred_train, y_test, ab_pred_test)

			
			if  round(train_accuracy) < treshold_2 :
				print (' The Model is Underfitting')
			

			elif round(train_accuracy - test_accuracy) > treshold_1 :
				print ('There is indicate Overfitting in the model')

				#need Tuning parameters 
				param = {'n_estimators': [100,300,500],'learning_rate': [0.1,0.05]}

				best_pred_train,best_pred_test=model_tuning(ab, param, X_train, y_train)

				#Evaluate Model 
				_,_,train_accuracy, test_accuracy = evaluate_model(y_train, best_pred_train, y_test, best_pred_test)

				if save_model is False :
					#Roc Curve
					ROC_CURVE(X_train,y_train,X_test,y_test,ab)
				elif save_model is True :
					filename  = filename
					save(ab,filename)
					#Roc Curve
					ROC_CURVE(X_train,y_train,X_test,y_test,ab)

			elif round(train_accuracy - test_accuracy) < treshold_1:
				
				if save_model is False :
					print('The model is good','\n')
					#Roc Curve
					ROC_CURVE(X_train,y_train,X_test,y_test,ab)
				elif save_model is True :
					print('The model is good','\n')
					filename  = filename
					save(ab,filename)
					#Roc Curve
					ROC_CURVE(X_train,y_train,X_test,y_test,ab)

				
################################# Gradient Boosting Classifier #################################

		if algorithm is 3 : 
			print('The algorithm = Gradient Boosting Classifier','\n')
			
			#model
			gbc,gbc_pred_train, gbc_pred_test=adaboost(X_train, X_test, y_train, y_test)

			#Evaluate Model 
			_,_,train_accuracy, test_accuracy = evaluate_model(y_train, gbc_pred_train, y_test, gbc_pred_test)

			
			if  round(train_accuracy) < treshold_2 :
				print (' The Model is Underfitting')

			elif round(train_accuracy - test_accuracy) > treshold_1 :
				print ('There is indicate Overfitting in the model')

				#need Tuning parameters 
				param = {'n_estimators': [100, 300,500],'learning_rate': [0.1,0.05],'loss':['deviance', 'exponential']}

				best_pred_train,best_pred_test=model_tuning(gbc, param, X_train, y_train)

				#Evaluate Model 
				_,_,train_accuracy, test_accuracy = evaluate_model(y_train, best_pred_train, y_test, best_pred_test)

				if save_model is False :
					#Roc Curve
					ROC_CURVE(X_train,y_train,X_test,y_test,gbc)
				elif save_model is True :
					filename  = filename
					save(gbc,filename)
					#Roc Curve
					ROC_CURVE(X_train,y_train,X_test,y_test,gbc)

			elif round(train_accuracy - test_accuracy) < treshold_1:
				
				if save_model is False :
					print('The model is good','\n')
					#Roc Curve
					ROC_CURVE(X_train,y_train,X_test,y_test,gbc)
				elif save_model is True :
					print('The model is good','\n')
					filename  = filename
					save(gbc,filename)
					#Roc Curve
					ROC_CURVE(X_train,y_train,X_test,y_test,gbc)


####################################################################################################
####################################### IF  IMBALANCE CLASS ########################################
	#train test split :
	elif imbalance is True :
		X = df.drop(['Survived'],axis = 1) 
		y = df['Survived']
		X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)
		# USING SMOTE
		sm = SMOTE(random_state=0)
		X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

		treshold_1 = 15
		treshold_2 = 70

################################# Random Forest Classifier #################################
		if algorithm is None :
			print('The algorithm = Random Forest Classifier','\n')
			
			#model
			rf,rf_pred_train, rf_pred_test=RandomForest(X_train_res, X_test, y_train_res, y_test)

			#Evaluate Model 
			_,_,train_accuracy, test_accuracy = evaluate_model(y_train_res, rf_pred_train, y_test, rf_pred_test)

			
			if  round(train_accuracy) < treshold_2 :
				print (' The Model is Underfitting')

			elif round(train_accuracy - test_accuracy) > treshold_1 :
				print ('There is indicate Overfitting in the model')

				#Tuning Parameters 
				param = {'n_estimators': [100, 300, 500],
				'max_features': [10,30,50],
				'max_depth': [10, 30, 50],
				'min_samples_split': [2, 5, 10],
				'min_samples_leaf': [10, 20, 30],
				'bootstrap': [True, False],
				'criterion': ['gini']}

				best_pred_train,best_pred_test=model_tuning(rf, param, X_train_res, y_train_res)

				#Evaluate Model 
				_,_,train_accuracy, test_accuracy = evaluate_model(y_train_res, best_pred_train, y_test, best_pred_test)

				if save_model is False :
					#Roc Curve
					ROC_CURVE(X_train_res,y_train_res,X_test,y_test,rf)
				elif save_model is True :
					filename  = filename
					save(rf,filename)
					#Roc Curve
					ROC_CURVE(X_train_res,y_train_res,X_test,y_test,rf)

			elif round(train_accuracy - test_accuracy) < treshold_1:
				
				if save_model is False :
					print('The model is good','\n')
					#Roc Curve
					ROC_CURVE(X_train_res,y_train_res,X_test,y_test,rf)
				elif save_model is True :
					print('The model is good','\n')
					filename  = filename
					save(rf,filename)
					#Roc Curve
					ROC_CURVE(X_train_res,y_train_res,X_test,y_test,rf)


################################# Bagging Classifier #################################

		if algorithm is 1 :
			print('The algorithm = Bagging Classifier (Base Estimator is DecisionTree)','\n')
			
			#model
			bg,bg_pred_train, bg_pred_test=Bagging(X_train_res, X_test, y_train_res, y_test)

			#Evaluate Model 
			_,_,train_accuracy, test_accuracy = evaluate_model(y_train_res, bg_pred_train, y_test, bg_pred_test)

			
			if  round(train_accuracy) < treshold_2 :
				print (' The Model is Underfitting')
			

			elif round(train_accuracy - test_accuracy) > treshold_1 :
				print ('There is indicate Overfitting in the model')

				#need Tuning parameters 
				param = {'n_estimators': [100, 300, 500],
				'max_features': [10,50],
				'max_samples' :[10,30],
				'bootstrap': [True, False]}

				best_pred_train,best_pred_test=model_tuning(bg, param, X_train_res, y_train_res)

				#Evaluate Model 
				_,_,train_accuracy, test_accuracy = evaluate_model(y_train_res, best_pred_train, y_test, best_pred_test)

				if save_model is False :
					#Roc Curve
					ROC_CURVE(X_train_res,y_train_res,X_test,y_test,bg)
				elif save_model is True :
					filename  = filename
					save(bg,filename)
					#Roc Curve
					ROC_CURVE(X_train_res,y_train_res,X_test,y_test,bg)

			elif round(train_accuracy - test_accuracy) < treshold_1:
				
				if save_model is False :
					print('The model is good','\n')
					#Roc Curve
					ROC_CURVE(X_train_res,y_train_res,X_test,y_test,bg)
				elif save_model is True :
					print('The model is good','\n')
					filename  = filename
					save(bg,filename)
					#Roc Curve
					ROC_CURVE(X_train_res,y_train_res,X_test,y_test,bg)

				
################################# Adaboost Classifier #################################

		if algorithm is 2 :
			print('The algorithm = Adaboost Classifier (Base Estimator is DecisionTree)','\n')
			
			#model
			ab,ab_pred_train, ab_pred_test=adaboost(X_train_res, X_test, y_train_res, y_test)

			#Evaluate Model 
			_,_,train_accuracy, test_accuracy = evaluate_model(y_train_res, ab_pred_train, y_test, ab_pred_test)

			
			if  round(train_accuracy) < treshold_2 :
				print (' The Model is Underfitting')
			

			elif round(train_accuracy - test_accuracy) > treshold_1 :
				print ('There is indicate Overfitting in the model')

				#need Tuning parameters 
				param = {'n_estimators': [100, 300, 500],'learning_rate': [0.1,0.05]}

				best_pred_train,best_pred_test=model_tuning(ab, param, X_train_res, y_train_res)

				#Evaluate Model 
				_,_,train_accuracy, test_accuracy = evaluate_model(y_train_res, best_pred_train, y_test, best_pred_test)

				if save_model is False :
					#Roc Curve
					ROC_CURVE(X_train_res,y_train_res,X_test,y_test,ab)
				elif save_model is True :
					filename  = filename
					save(ab,filename)
					#Roc Curve
					ROC_CURVE(X_train_res,y_train_res,X_test,y_test,ab)

			elif round(train_accuracy - test_accuracy) < treshold_1:
				
				if save_model is False :
					print('The model is good','\n')
					#Roc Curve
					ROC_CURVE(X_train_res,y_train_res,X_test,y_test,ab)
				elif save_model is True :
					print('The model is good','\n')
					filename  = filename
					save(ab,filename)
					#Roc Curve
					ROC_CURVE(X_train_res,y_train_res,X_test,y_test,ab)

				
################################# Gradient Boosting Classifier #################################

		if algorithm is 3 : 
			print('The algorithm = Gradient Boosting Classifier','\n')
			
			#model
			gbc,gbc_pred_train, gbc_pred_test=adaboost(X_train_res, X_test, y_train_res, y_test)

			#Evaluate Model 
			_,_,train_accuracy, test_accuracy = evaluate_model(y_train_res, gbc_pred_train, y_test, gbc_pred_test)

			
			if  round(train_accuracy) < treshold_2 :
				print (' The Model is Underfitting')

			elif round(train_accuracy - test_accuracy) > treshold_1 :
				print ('There is indicate Overfitting in the model')

				#need Tuning parameters 
				param = {'n_estimators': [100, 300, 500],'learning_rate': [0.1,0.05],'loss':['deviance', 'exponential']}

				best_pred_train,best_pred_test=model_tuning(gbc, param, X_train_res, y_train_res)

				#Evaluate Model 
				_,_,train_accuracy, test_accuracy = evaluate_model(y_train_res, best_pred_train, y_test, best_pred_test)

				if save_model is False :
					#ROC Curve
					ROC_CURVE(X_train_res,y_train_res,X_test,y_test,gbc)
				elif save_model is True :
					filename  = filename
					save(gbc,filename)
					#Roc Curve
					ROC_CURVE(X_train_res,y_train_res,X_test,y_test,gbc)

			elif round(train_accuracy - test_accuracy) < treshold_1:
				
				if save_model is False :
					print('The model is good','\n')
					#Roc Curve
					ROC_CURVE(X_train_res,y_train_res,X_test,y_test,gbc)
				elif save_model is True :
					print('The model is good','\n')
					filename  = filename
					save(gbc,filename)
					#Roc Curve
					ROC_CURVE(X_train_res,y_train_res,X_test,y_test,gbc)

	