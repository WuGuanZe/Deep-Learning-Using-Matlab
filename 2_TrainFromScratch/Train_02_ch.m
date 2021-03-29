%% 創建小型深度學習網路進行手寫數字分類-2

%% 載入影像資料
digitDatasetPath01 = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');

digitData01 = imageDatastore(digitDatasetPath01, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% 從資料庫中顯示影像
figure;
perm = randperm(10000, 20);
for i = 1:20
    subplot(4,5,i);
    img01 = readimage(digitData01, perm(i));
    imshow(img01);
end

%% 確認每個分類中的影像數量
CountLabel01 = digitData01.countEachLabel

%%  切割訓練與測試資料
trainingNumFiles01 = 750;
[trainDigitData01,testDigitData01] = splitEachLabel(digitData01, ...
    trainingNumFiles01, 'randomize');

%% 定義網路架構
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
    
%     maxPooling2dLayer(2,'Stride',2)
%     
%     convolution2dLayer(3,64,'Padding',1)
%     batchNormalizationLayer
%     reluLayer
%     
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%% 設定訓練參數
options = trainingOptions(...
    'sgdm',...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128,...
    'InitialLearnRate', 0.01,...
    'ExecutionEnvironment', 'auto',...
    'Plots', 'training-progress',...
    'ValidationData', testDigitData01,...
    'ValidationFrequency', 30);

%% 訓練網路
convnet01 = trainNetwork(trainDigitData01, layers, options);

%% 在測試影像中進行影像分類
predictedLabels01  = classify(convnet01, testDigitData01);
valLabels01  = testDigitData01.Labels;

%% 計算精準度
accuracy = sum(predictedLabels01 == valLabels01)/numel(valLabels01)

%% 計算混淆矩陣
figure
[cmat,classNames] = confusionmat(valLabels01,predictedLabels01);
h = heatmap(classNames,classNames,cmat);
xlabel('Predicted Class');
ylabel('True Class');
title('Confusion Matrix');