# PointTransformer for Point Cloud Processing

***

# Introduction
This is PyTorch implementation for the training and testing of the PoinTransformer architecture for the semantic segmentation of automotive Radar Point Cloud Dataset.

This research presents a comprehensive study on the semantic segmentation of automotive radar point clouds, leveraging two state-of-the-art deep learning models namely PointNet++ and PointTransformer. The experiments were conducted on three distinct configurations of the nuScenes radar dataset, each representing a unique scenario in autonomous driving. PointNet++ was chosen for its proven performance in radar-based semantic segmentation, while PointTransformer, a relatively recent model in the domain, was introduced to explore its potential for Radar point cloud-based deep learning application in autonomous driving. The implementation, training, and evaluation of both models were carried out, and the results were compared based on the class-wise F1 scores and the overall macro-averaged F1 score.

Ablation studies were also performed on PointTransformer, investigating the influence of various hyperparameters and the influence of di↵erent radar features, revealing insights into the key factors a↵ecting model performance.
