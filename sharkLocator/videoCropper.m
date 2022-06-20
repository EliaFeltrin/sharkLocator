% detect sharks from video. Extract framePerSec frames per second
framePerSec = 1/15;

% trainedPath = ".....";
% trained = load(trainedPath);
%detector = trained.detector;
%detector = yolov4ObjectDetector("csp-darknet53-coco");

videosToDetectDir = strcat(pwd, "/toDetect/videos/"); 
videos = dir(videosToDetectDir);
varTypes = ["string", "single","single", "single", "string"];
varNames = ["fileName", "frameNumber", "atTime", "score", "originalVideo"];
croppedTable = table('VariableNames',varNames,'VariableTypes', varTypes,'Size', [1, 5]);

croppedImagesDir = strcat(pwd, "/sharkLocator/cropped/");

counter = 0;

for i = 3 : size(videos, 1)
    video = videos(i);
    V = VideoReader(strcat(videosToDetectDir, video.name));

    for k = 1 : floor(V.NumFrames * framePerSec / V.FrameRate)
        
        I = read(V, floor(V.FrameRate * k / framePerSec));

        [bboxes, scores] = detect(detector, I);

        for j = 1:size(bboxes, 1)
            counter = counter + 1;
            vidName = strcat(croppedImagesDir, "v", int2str(k), "_", int2str(j), ".jpg");
            J = imcrop(I, bboxes(j,:));
            croppedTable(counter, "fileName")       = {vidName};
            croppedTable(counter, "frameNumber")    = {floor(V.FrameRate * k / framePerSec)};
            croppedTable(counter, "atTime")         = {k / framePerSec};
            croppedTable(counter, "score")          = {scores(j)};
            croppedTable(counter, "originalVideo")  = {strcat(videosToDetectDir, video.name)};
            imwrite(J, vidName);
        end

    end
    

end