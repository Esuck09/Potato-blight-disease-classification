clc

imds = imageDatastore('potato',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

%%
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);


%%
net = inceptionv3

%%
numClasses = numel(categories(imdsTrain.Labels))

%%
myNet = layerGraph(net);

myNet = replaceLayer(myNet, 'predictions', [fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)]);

myNet = replaceLayer(myNet, 'predictions_softmax', softmaxLayer);

myNet = replaceLayer(myNet, 'ClassificationLayer_predictions', classificationLayer);

%%
inputSize = myNet.Layers(1).InputSize;

%%
layersTransfer = myNet.Layers(1:end-3);
%%
clc
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

%%
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'ExecutionEnvironment', 'multi-gpu',...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

%%
clc
TrainedInception = trainNetwork(augimdsTrain,myNet,options);


%%
[YPred,scores] = classify(netTransfer,augimdsValidation)

%%
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)


%%
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