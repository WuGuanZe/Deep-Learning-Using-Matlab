%% �Ыؤp���`�׾ǲߺ����i���g�Ʀr����-1

%% ���J�v�����
digitDatasetPath01 = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');

digitData01 = imageDatastore(digitDatasetPath01, ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% �q��Ʈw����ܼv��
figure;
perm01 = randperm(10000, 20);
for i = 1:20
    subplot(4,5,i);
    img = readimage(digitData01, perm01(i));
    imshow(img);
end

%% �T�{�C�Ӥ��������v���ƶq
CountLabel = digitData01.countEachLabel

%% ���ΰV�m�P���ո��
trainingNumFiles01 = 750;
[trainDigitData01,testDigitData01] = splitEachLabel(digitData01, ...
    trainingNumFiles01, 'randomize');

%% �w�q�����[�c
% �o��Х�Deepnetwork design�ԥX�@�ӻP�U��@�˪��ҫ�
% layers = [
%     imageInputLayer([28 28 1])
%     convolution2dLayer(5, 20)
%     reluLayer
%     maxPooling2dLayer(2, 'Stride', 2)
%     fullyConnectedLayer(10)
%     softmaxLayer
%     classificationLayer];

%% �]�w�V�m�Ѽ�
options = trainingOptions(...
    'sgdm',...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 128,...
    'InitialLearnRate', 0.0001,...
    'ExecutionEnvironment', 'auto',...
    'Plots', 'training-progress');

%% �V�m����
convnet01 = trainNetwork(trainDigitData01, layers_1, options);

%% �b���ռv�����i��v������
predictedLabels01  = classify(convnet01, testDigitData01);
valLabels01  = testDigitData01.Labels;

%% �p���ǫ�
accuracy = sum(predictedLabels01 == valLabels01)/numel(valLabels01)
