%% Confusion matrix
[YPred,scores] = classify(TrainedInception,augimdsValidation)

%% try and do the f1 for each of the classes and compare
YValidation = imdsValidation.Labels;

accuracy = mean(YPred == YValidation)

cm = confusionmat(YValidation,YPred);
cmt = cm'

%% calculation for F1 score
diagonal = diag(cmt)
sum_of_rows = sum(cmt, 2)

precision = diagonal ./sum_of_rows
avg_precision = mean(precision)


sum_of_columns = sum(cmt, 1)

recall = diagonal ./ sum_of_columns'
avg_recall = mean(recall)

f1_score = 2*((avg_precision*avg_recall) / (avg_precision+avg_recall))

%% Visualise confusion matrix
figure;
imshow(cm, 'InitialMagnification', 'fit');
colorbar;
title('Confusion Matrix');
xlabel('Predicted Class');
ylabel('True Class');
