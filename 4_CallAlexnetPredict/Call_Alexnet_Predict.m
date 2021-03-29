img01 = imread('peppers.png');
img02 = imresize(img01,[227 227]);
out01 = Alexnet_Predict(img02);

[scores01, index01] = sort(out01, 'descend');
net01 = alexnet;
classnet01 = net01.Layers(end).Classes;
classname01 = net01.Layers(end).Classes;
classname01(index01(1:3))