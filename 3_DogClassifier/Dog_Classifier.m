img01 = imread('Demo01.jpg');
img02 = imresize(img01,[227 227]);
classify(trainedNetwork_1, img02)
