clc

imds = imageDatastore('potato',...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

%%
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);


%%
net = resnet18

%%
numClasses = numel(categories(imdsTrain.Labels))

%%
myNet = layerGraph(net);

myNet = replaceLayer(myNet, 'fc1000', [fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)]);

myNet = replaceLayer(myNet, 'prob', softmaxLayer);

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
TrainedResNet = trainNetwork(augimdsTrain,myNet,options);
