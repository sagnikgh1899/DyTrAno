def get_outlier_inlier_info_post_filtering(data, labels):
    label_list = []
    anomaly_list = []
    max_val = float("-inf")
    for key in labels:
        if max_val < labels[key]:
            max_val = labels[key]
        if labels[key] == -1:
            anomaly_list.append(data[key])
        else:
            label_list.append(labels[key])
    cluster_num = max_val
    print("\n------------ After Outlier Post Processing ------------")
    print("Total number of data points =", len(data))
    print("Number of Non-Anomalies =", len(label_list))
    print("Number of Anomalies =", len(anomaly_list))
    print("Number of Cluster(s) =", cluster_num)
