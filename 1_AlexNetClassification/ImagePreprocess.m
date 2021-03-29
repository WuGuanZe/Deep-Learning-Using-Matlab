function OutImage= preprocessImg(filename)

Image= imread(filename);

OutImage = imresize(Image, [227,227]);

end

