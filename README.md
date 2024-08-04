## Dataset:https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset/data

## Followed Notebook:https://www.kaggle.com/code/atharvaingle/plant-disease-classification-resnet-99-2

CUSTOM RESNET9 MODEL RESULTS

Confusion Matrix:
![Figure_1](https://github.com/user-attachments/assets/30f954d2-9d7b-4c30-8677-f5da2ad714dd)


Test scores Accuracy(33 images): 100.00%

Validation metrics

![Figure_2](https://github.com/user-attachments/assets/cfb31138-90e0-4946-81cd-a7a5ddcf697d)

Loss: 0.0285

Accuracy: 99.19%

Precision: 0.9920

Recall: 0.9919

F1 Score: 0.9918

## CUSTOM RESNET9 MODEL WITH SVM AS A CLASSIFIER RESULTS

![image](https://github.com/user-attachments/assets/317ccb91-94ff-4ec0-97d2-29116711f75f)

### From this ResNet architecture you need to remove the last classifer and instead use the extracted features to train the SVM.

Confusion Matrix:
![Figure_1svm](https://github.com/user-attachments/assets/2a2b2c44-5677-430a-8da0-e296d6f2f812)

![Figure_2SVM](https://github.com/user-attachments/assets/4649579f-ea63-435e-a6f0-77e641d162cd)

Accuracy: 99.71%

Precision: 0.9971

Recall: 0.9971

F1 Score: 0.9971


## CUSTOM RESNET9 MODEL WITH RANDOM FOREST AS A CLASSIFIER RESULTS

![Figure_1RF](https://github.com/user-attachments/assets/d59d837b-68a0-40b4-a997-1b93d1cfcf3c)

![Figure_2RF](https://github.com/user-attachments/assets/d0edd6a4-7a1c-4aa6-97b6-2f11410a1764)

Accuracy: 99.13%

Precision: 0.9914

Recall: 0.9913

F1 Score: 0.9913

## CUSTOM RESNET9 MODEL WITH KNN AS A CLASSIFIER RESULTS

![Figure_1KNN](https://github.com/user-attachments/assets/47d5c40c-239a-482f-bd3f-de2772109233)


![Figure_2KNN](https://github.com/user-attachments/assets/96805865-30f5-46c8-ba3f-8b274f931eb4)

Accuracy: 99.64%

Precision: 0.9964

Recall: 0.9964

F1 Score: 0.9964

##  PATCH EMBEDDING 

![1_CcjZmfOrzU6mDlTljvrqHw](https://github.com/user-attachments/assets/e9c81a23-c3d1-431d-9061-e7eae7f6f4c3)


Patch Extraction: The PatchEmbedding class uses a nn.Conv2d layer to divide the input image into non-overlapping patches (patch_size) and projects them into an embedding space of size emb_size.

Flattening and Reshaping: The extracted patches are flattened and transposed to form a sequence of patch embeddings suitable for subsequent layers.

Val_accuracy: 0.9855







