%% 創建小型深度學習網路進行手寫數字分類-1

%% 載入影像資料
digitDatasetPath01 = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');

digitData01 = imageDatastore(digitDatasetPath01, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% 從資料庫中顯示影像
figure;
perm01 = randperm(10000, 20);
for i = 1:20
    subplot(4,5,i);
    img = readimage(digitData01, perm01(i));
    imshow(img);
end

%% 確認每個分類中的影像數量
CountLabel = digitData01.countEachLabel

%% 切割訓練與測試資料
trainingNumFiles01 = 750;
[trainDigitData01,testDigitData01] = splitEachLabel(digitData01, ...
    trainingNumFiles01, 'randomize');

%% 定義網路架構
% 這邊請用Deepnetwork design拉出一個與下方一樣的模型
% layers = [
%     imageInputLayer([28 28 1])
%     convolution2dLayer(5, 20)
%     reluLayer
%     maxPooling2dLayer(2, 'Stride', 2)
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];

%% 設定訓練參數
options = trainingOptions(...
    'sgdm',...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128,...
    'InitialLearnRate', 0.0001,...
    'ExecutionEnvironment', 'auto',...
    'Plots', 'training-progress');

%% 訓練網路
convnet01 = trainNetwork(trainDigitData01, layers_1, options);

%% 在測試影像中進行影像分類
predictedLabels01  = classify(convnet01, testDigitData01);
valLabels01  = testDigitData01.Labels;

%% 計算精準度
accuracy = sum(predictedLabels01 == valLabels01)/numel(valLabels01)
