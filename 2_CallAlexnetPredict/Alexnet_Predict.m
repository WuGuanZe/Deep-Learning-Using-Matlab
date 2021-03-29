function out01 = Alexnet_Predict(img01)

persistent net01;

if isempty(net01)
    net01 = coder.loadDeepLearningNetwork('alexnet');
end

out01= predict(net01, img01);
end
