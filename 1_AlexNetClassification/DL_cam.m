net = resnet50;
cam  = webcam;

while(1)
    img = snapshot(cam);
    image(img)
    img2 = imresize(img,[224 224]);
    label = classify(net,img2);
    disp(label)
end

