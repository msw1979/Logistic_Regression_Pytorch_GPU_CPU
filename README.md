# Logistic Regression Pytorch GPU CPU
This is an example of Logistic Regression using Pytorch (GPU or CPU). The data used here is the Chrun data and can be downloaded from: https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv. The code split the data to training and validation data sets and normalize the feature part, then load it to training loaders. The user has the choice to use the subclass and sequential model. The code will generate loss and accuracy data and plot them. It also plot confusion matrix plots for training and validation data. 
The figure below show the loss and accuracy versus epoch for training and validation process:

![loss_accuracy_epoch](https://user-images.githubusercontent.com/12114448/222920512-ee430971-84b5-43c2-a845-9e956ca181a0.png)


The figures below show the confusion matrix for training and validation data:

![confusion_matrix_test](https://user-images.githubusercontent.com/12114448/222920523-2eae994c-42c8-4685-98dd-0d910f1fa576.png)
![confusion_matrix_train](https://user-images.githubusercontent.com/12114448/222920525-fe254a7c-3fa6-4704-b272-d46f7ae37483.png)
