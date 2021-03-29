% alexnet
net = alexnet;

img01 = imread('cat.jpg');
figure,imshow(img01);

img02 = imresize(img01,[227 227]);
label01 = classify(net,img02)