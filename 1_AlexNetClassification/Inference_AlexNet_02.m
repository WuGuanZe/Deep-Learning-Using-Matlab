clear all,close all,clc
%% ImageDatastore
net = alexnet;
net.Layers

%% Show what AlexNet does with random images without being retrained
examples01 = imageDatastore('ExampleImages',...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

%% Count files in ImageDatastore labels
countEachLabel(examples01)

%% Split ImageDatastore labels by proportions
examplespart01 = splitEachLabel(examples01, 2);
countEachLabel(examplespart01)
 
%% Change ReadFunction in imageDatastore 
% Resize image before reading it
examplespart01.ReadFcn = @ImagePreprocess;
img01 = readimage(examplespart01,5);
%whos img

% Make prediction
classLabel = classify(net, img01);

% Show image
imshow(img01); 
title(char(classLabel));
