from model import DonutX
import pandas as pd
import numpy as np
from kpi_series import KPISeries
from sklearn.metrics import precision_recall_curve
from evaluation_metric import range_lift_with_delay



# ---- THIS IS OUR CODE ----

# read train and test data
df = pd.read_csv('../train.csv', header=0, index_col=None)
df_2 = pd.read_csv('../test.csv', header=0, index_col=None)

# get the list of KPIs
kpi_list = df_2["KPI ID"].unique().tolist()
# print(kpi_list)

# list of final kpi data
appended_data = []

# iterate through all the KPIs
for curr_kpi in kpi_list:
	print("-----------------------")
	print("CURRENT KPI", curr_kpi)

	# get a df of current kpi for training - we only want timestamp, value and label columns
	df_train = df[df["KPI ID"] == curr_kpi]
	df_train = df_train[["timestamp", "value", "label"]]
	# print(df_train)

	# split into train and valid samples: e.g. 80%:20% ratio
	# NOTE: sample data split the data in the KPI data structure
	df_train, df_valid = np.split(df_train, [int(.7*len(df_train))])

	# get a df of current kpi for testing - we only want timestamp, value columns
	df_test = df_2[df_2["KPI ID"] == curr_kpi]
	df_test = df_test[["timestamp", "value"]]
	# print(df_test)

	# create a train KPI data structure. NOTE: it fills in missing points
	train_kpi = KPISeries(
	    value = df_train.value,
	    timestamp = df_train.timestamp,
	    label = df_train.label,
	)

	# create a valid KPI data structure NOTE: it fills in missing points
	valid_kpi = KPISeries(
	    value = df_valid.value,
	    timestamp = df_valid.timestamp,
	    label = df_valid.label,
	)

	# create a test KPI data structure
	# NOTE: it fills in missing points AND creates a label columns of 0s
	test_kpi = KPISeries(
	    value = df_test.value,
	    timestamp = df_test.timestamp,
	)

	print(np.unique(train_kpi.label, return_counts=True))
	print(np.unique(valid_kpi.label, return_counts=True))
	print(np.unique(test_kpi.label, return_counts=True))
	print(np.unique(test_kpi.missing, return_counts=True))

	# normalise the data, get the mean and standard deviation
	train_kpi, train_kpi_mean, train_kpi_std = train_kpi.normalize(return_statistic=True)
	valid_kpi = valid_kpi.normalize(mean=train_kpi_mean, std=train_kpi_std)
	test_kpi = test_kpi.normalize(mean=train_kpi_mean, std=train_kpi_std)

	# create the model. There is also a Donut model (not DonutX)
	model = DonutX(cuda=False, max_epoch=50, latent_dims=8, network_size=[100, 100])

	# TRAIN THE MODEL using train_kpi and valid_kpi
	# NOTE: there is also a label_sampling method (i think for supervised/unsupervised)
	print("***start fitting")
	model.fit(train_kpi.label_sampling(1.), valid_kpi)
	# model.fit(train_kpi, valid_kpi)
	print("***fitting complete")

	# USE THE MODEL TO PREDICT
	print("***start predicting")
	# predicted_labels, threshold = model.detect(test_kpi, return_threshold=True)
	y_prob = model.predict(test_kpi.label_sampling(0.))
	# y_prob, predicted_labels, threshold = model.detect(test_kpi.label_sampling(0.), return_threshold=True)
	print("***start predicting")

	# find the f1-scores and get the threshold for the highest f1-score
	precisions, recalls, thresholds = precision_recall_curve(test_kpi.label, y_prob)
	f1_scores = (2 * precisions * recalls) / (precisions + recalls)
	print(f'best F1-score: {np.max(f1_scores[np.isfinite(f1_scores)])}')
	thre = thresholds[np.argmax(f1_scores)]
	print('best threshold: ', thre)

	# find predicted values
	pred = y_prob >= thre
	print("anomaly counts:", np.unique(pred, return_counts=True))

	# print output of prediction
	# print("predicted labels, threshold: ", predicted_labels, threshold)
	# unique, counts = np.unique(predicted_labels, return_counts=True)
	# print("anomaly counts: ", unique, counts)

	# add the predicted labels to the test KPI 
	test_kpi._label = np.asarray(pred, np.int).astype(int)


	# format results for outputting
	final_df = pd.DataFrame()
	final_df["timestamp"] = test_kpi.timestamp
	final_df["KPI ID"] = curr_kpi		#this can't be first because dataframe starts empty
	final_df["predict"] = test_kpi.label
	final_df["missing"] = test_kpi.missing
	# remove rows that were missing
	final_df = final_df[final_df.missing != 1]
	# remove missing column
	final_df = final_df.drop('missing', axis=1)
	# rearrange columns for formatting
	final_df = final_df.reindex(columns = ["KPI ID", "timestamp", "predict"])

	# append to final results
	appended_data.append(final_df)



# concat results
appended_data = pd.concat(appended_data)

# dump results to file
appended_data.to_csv('./testsubmission3.csv', index=False)

# check the total anomaly count
print("total anomaly count: ")
print(np.unique(appended_data.predict, return_counts=True))
