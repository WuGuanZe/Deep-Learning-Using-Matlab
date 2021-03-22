%% �Ыؤp���`�׾ǲߺ����i���g�Ʀr����-2

%% ���J�v�����
digitDatasetPath01 = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');

digitData01 = imageDatastore(digitDatasetPath01, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% �q��Ʈw����ܼv��
figure;
perm = randperm(10000, 20);
for i = 1:20
    subplot(4,5,i);
    img01 = readimage(digitData01, perm(i));
    imshow(img01);
end

%% �T�{�C�Ӥ��������v���ƶq
CountLabel01 = digitData01.countEachLabel

%%  ���ΰV�m�P���ո��
trainingNumFiles01 = 750;
[trainDigitData01,testDigitData01] = splitEachLabel(digitData01, ...
    trainingNumFiles01, 'randomize');

%% �w�q�����[�c
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

%% �]�w�V�m�Ѽ�
options = trainingOptions(...
    'sgdm',...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128,...
    'InitialLearnRate', 0.01,...
    'ExecutionEnvironment', 'auto',...
    'Plots', 'training-progress',...
    'ValidationData', testDigitData01,...
    'ValidationFrequency', 30);

%% �V�m����
convnet01 = trainNetwork(trainDigitData01, layers, options);

%% �b���ռv�����i��v������
predictedLabels01  = classify(convnet01, testDigitData01);
valLabels01  = testDigitData01.Labels;

%% �p���ǫ�
accuracy = sum(predictedLabels01 == valLabels01)/numel(valLabels01)

%% �p��V�c�x�}
figure
[cmat,classNames] = confusionmat(valLabels01,predictedLabels01);
h = heatmap(classNames,classNames,cmat);
xlabel('Predicted Class');
ylabel('True Class');
title('Confusion Matrix');