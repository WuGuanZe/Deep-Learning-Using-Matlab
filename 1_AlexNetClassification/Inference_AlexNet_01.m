%% Use alexnet to do inference
% Load Pre-trained CNN
net = alexnet;
% Show the architecture of AlexNet
net.Layers
%% Classify 'peppers' in 4 lines of code
% Import a testing image.
img01 = imread('cat.jpg');

% There is a size requirement of 227 x 227 for AlexNet. 
img02 = imresize(img01, [227 227]);

% Recognize the testing image
[Ypred, scores] = classify(net, img02);

% Show predicting result
imshow(img02);
title(char(Ypred))

%% List top 3 class scores
[ssort, sidx] = sort(scores, 'descend');

numTopClasses = 3; % show top N choices
TopClasses = net.Layers(end).ClassNames(sidx(1:numTopClasses));
TopScores = ssort(1:numTopClasses)';

topTable = table(TopClasses, TopScores)

