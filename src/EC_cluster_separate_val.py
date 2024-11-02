import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
import argparse
warnings.filterwarnings("ignore", category=FutureWarning)

def calc_dist(x, y):
  return np.sqrt(np.sum((x - y)**2))
  
def main(args):

    total_subjects = args.total_subjects

    dists_from_clusters = {"cluster_1":[], "cluster_2":[], "prob_cluster_1":[], "prob_cluster_2":[]}

    for validation in range(0,total_subjects):
    
      df_dict_new = torch.load(args.input_filepath + "_train_" + str(validation) + ".pt")

      if args.feat_type == "bottom_50_percent_sd":
        key_to_drop = list(df_dict_new.drop('labels', axis=1).std().sort_values(ascending=True).keys()[args.total_feats//2:])
      elif args.feat_type == "top_50_percent_mean":
        key_to_drop = list(df_dict_new.drop('labels', axis=1).mean().sort_values(ascending=True).keys()[:args.total_feats//2])
      elif all_vals:
        key_to_drop = []
      elif args.feat_type == "top_50_percent_sd":
        key_to_drop = list(df_dict_new.drop('labels', axis=1).std().sort_values(ascending=True).keys()[:args.total_feats//2])
      else:
        raise ValueError("Invalid Choice")

      for kk in key_to_drop:
        if kk != 'labels':
          df_dict_new = df_dict_new.drop(kk, axis=1)

      df_dict_validation = torch.load(args.input_filepath + "_valid_" + str(validation) + ".pt").drop('labels', axis=1)
      df_dict_all_std = df_dict_new.drop('labels', axis=1)
      mean_val = df_dict_all_std.mean()
      std_val = df_dict_all_std.std()
      df_dict_all_std = (df_dict_all_std - mean_val) / std_val

      df_dict_validation = (df_dict_validation - mean_val) / std_val
      
      pca = PCA(n_components=args.components)
      df_dict_all_pca = pca.fit_transform(df_dict_all_std)

      df_dict_validation_pca = pca.transform(df_dict_validation)

      kmeans = KMeans(n_clusters=2)
      kmeans.fit(df_dict_all_pca)

      class_0_indices = np.where(np.asarray(kmeans.labels_) == 0)[0]
      class_1_indices = np.where(np.asarray(kmeans.labels_) == 1)[0]

      new_labels = list(df_dict_new.labels)
      target = new_labels

      dmax = np.max(np.linalg.norm(df_dict_all_pca - df_dict_all_pca[:,None], axis=-1))

      if np.abs(df_dict_all_pca[np.array(np.where(np.array(new_labels) == 0)[0]),0].mean() - kmeans.cluster_centers_[0,0]) < np.abs(df_dict_all_pca[np.array(np.where(np.array(new_labels) == 1)[0]),0].mean() - kmeans.cluster_centers_[0,0]):
        center_1 = kmeans.cluster_centers_[0,:]
        center_2 = kmeans.cluster_centers_[1,:]
      else:
        center_1 = kmeans.cluster_centers_[1,:]
        center_2 = kmeans.cluster_centers_[0,:]

      dists_from_clusters["cluster_1"].append(1 - calc_dist(df_dict_validation_pca, center_1)/dmax)
      dists_from_clusters["cluster_2"].append(1 - calc_dist(df_dict_validation_pca, center_2)/dmax)
      dd1 = (dists_from_clusters["cluster_1"][-1])/((dists_from_clusters["cluster_1"][-1]) + (dists_from_clusters["cluster_2"][-1]))

      dists_from_clusters["prob_cluster_1"].append(dd1)
      dists_from_clusters["prob_cluster_2"].append(1 - dd1)

    prob_class_1, prob_class_2 = dists_from_clusters["prob_cluster_1"], dists_from_clusters["prob_cluster_2"]
    preds = np.stack((np.array(prob_class_1), np.array(prob_class_2)), axis=1)
    overall_acc = 100*np.sum(np.argmax(preds, axis=1) == target_labels)/len(target_labels)
    acc_preds = np.int32(np.argmax(preds, axis=1) == target_labels)
    OCMI = 100*np.sum(np.array([preds[i, np.argmax(preds, axis=1)[i]] for i in range(len(np.argmax(preds, axis=1)))])*acc_preds)/len(target_labels)
    
    return overall_acc, OCMI
    
 if __name__ == 'main':
    parser = argparse.ArgumentParser(description='EC Cluster')
    parser.add_argument('--input_filepath', type=str)
    parser.add_argument('--total_subjects', type=int)
    parser.add_argument('--total_feats', type=int)
    parser.add_argument('--components', type=int)
    parser.add_argument('--feat_type', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--save_op', default=False, type=bool)
    args = parser.parse_args()
    main(args)