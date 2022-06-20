% detect sharks from image and create new images cropped from original

% trainedPath = ".....";
% trained = load(trainedPath);
%detector = trained.detector;
detector = yolov4ObjectDetector("csp-darknet53-coco");

imagesToDetectDir = strcat(pwd, "/toDetect/images/"); 
images = dir(imagesToDetectDir);
varTypes = ["string", "single", "string"];
varNames = ["fileName", "score", "originalImage"];
croppedTable = table('VariableNames',varNames,'VariableTypes', varTypes,'Size', [size(images,1)-2, 3]);

croppedImagesDir = strcat(pwd, "/sharkLocator/cropped/");

counter = 0;
for i = 3 : size(images)
    image = images(i);
    I = imread(strcat(imagesToDetectDir, image.name));
    [bboxes, scores] = detect(detector, I);

    for j = 1:size(bboxes, 1)
        counter = counter + 1;
        imName = strcat(croppedImagesDir, "i", int2str(counter), "_", int2str(j), ".jpg");
        J = imcrop(I, bboxes(j,:));
        croppedTable(counter, "fileName") = {imName};
        croppedTable(counter, "score") = {scores(j)};
        croppedTable(counter, "originalImage") = {strcat(imagesToDetectDir, image.name)};
        imwrite(J, imName);
    end

end