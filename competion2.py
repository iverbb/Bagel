from model import DonutX
import pandas as pd
import numpy as np
from kpi_series import KPISeries
from sklearn.metrics import precision_recall_curve
from evaluation_metric import range_lift_with_delay

df = pd.read_csv('../train.csv', header=0, index_col=None)
df_2 = pd.read_csv('../test.csv', header=0, index_col=None)

kpi_list = df["KPI ID"].unique().tolist()

appended_data = []

for curr_kpi in kpi_list:
	print("--------------------")
	print("CURRENT KPI: ", curr_kpi)

	df_train = df[df["KPI ID"] == curr_kpi]
	df_train = df_train[["timestamp", "value", "label"]]

	df_predict = df_2[df_2["KPI ID"] == curr_kpi]
	df_predict = df_predict[["timestamp", "value"]]

	# df_train = pd.read_csv('./sample_data.csv' , header=0, index_col=None)

	print(df_train)

	kpi = KPISeries(
	    value = df_train.value,
	    timestamp = df_train.timestamp,
	    label = df_train.label,
	    name = 'sample_data',
	)



	train_kpi, valid_kpi, test_kpi = kpi.split((0.49, 0.21, 0.3))
	train_kpi, train_kpi_mean, train_kpi_std = train_kpi.normalize(return_statistic=True)
	valid_kpi = valid_kpi.normalize(mean=train_kpi_mean, std=train_kpi_std)
	test_kpi = test_kpi.normalize(mean=train_kpi_mean, std=train_kpi_std)

	print(np.unique(train_kpi.label, return_counts=True))
	print(np.unique(valid_kpi.label, return_counts=True))
	print(np.unique(test_kpi.label, return_counts=True))
	print(np.unique(test_kpi.missing, return_counts=True))

	model = DonutX(cuda=False, max_epoch=50, latent_dims=8, network_size=[100, 100])
	# model.fit(train_kpi.label_sampling(0.), valid_kpi)
	model.fit(train_kpi.label_sampling(1.), valid_kpi)
	y_prob = model.predict(test_kpi.label_sampling(0.))
	# y_prob, pred, thre = model.detect(test_kpi.label_sampling(0.), return_threshold=True)

	print("y prob before: ", y_prob)
	print("length before: ", len(y_prob))

	y_prob = range_lift_with_delay(y_prob, test_kpi.label)

	print("LENGTHS: ", len(y_prob), len(test_kpi.label))
	precisions, recalls, thresholds = precision_recall_curve(test_kpi.label, y_prob)
	f1_scores = (2 * precisions * recalls) / (precisions + recalls)
	print(f'best F1-score: {np.max(f1_scores[np.isfinite(f1_scores)])}')
	print('best threshold: ', thresholds[np.argmax(f1_scores)])

	print("xxxxxxx")
	print(f1_scores)
	print(len(f1_scores))

	print("****")
	print("thresholds: ", thresholds)
	print(len(precisions), len(recalls), len(thresholds))

	print("----")
	print(y_prob)
	print("max value y_prob: ", np.max(y_prob[np.isfinite(y_prob)]))
	# print(pred)
	# print(thre)
	# print(np.unique(pred, return_counts=True))	# predicted anomaly counts

	print('DDDDDDD')
	thre = thresholds[np.argmax(f1_scores)]
	pred = y_prob >= thre
	print(np.unique(pred, return_counts=True))




	print()
	print("NOW DO SOME PREDICTING")


	print(df_predict)

	predict_kpi = KPISeries(
	    value = df_predict.value,
	    timestamp = df_predict.timestamp,
	    name = 'predict_data',
	)

	print(np.unique(predict_kpi.label, return_counts=True))
	print(np.unique(predict_kpi.missing, return_counts=True))

	predict_kpi = predict_kpi.normalize(mean=train_kpi_mean, std=train_kpi_std)

	z_prob = model.predict(predict_kpi.label_sampling(0.))

	print("y prob before: ", z_prob)
	print("length of z probs: ", len(z_prob))
	# y_prob = range_lift_with_delay(z_prob, test_kpi.label)
	# print("LENGTHS: ", len(z_prob), len(test_kpi.label))


	anomalies = z_prob >= thre
	print("anomaly counts:", np.unique(anomalies, return_counts=True))

	predict_kpi._label = np.asarray(anomalies, np.int).astype(int)


	# format results for outputting
	final_df = pd.DataFrame()
	final_df["timestamp"] = predict_kpi.timestamp
	final_df["KPI ID"] = curr_kpi		#this can't be first because dataframe starts empty
	final_df["predict"] = predict_kpi.label
	final_df["missing"] = predict_kpi.missing
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
appended_data.to_csv('./testsubmission4.csv', index=False)

# check the total anomaly count
print("total anomaly count: ")
print(np.unique(appended_data.predict, return_counts=True))
