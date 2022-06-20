% code per detection partendo da examples in https://github.com/zacqoo/tf_shark_detector_colab

clear
clc

%% params
% dataset image load to trainin dataset
nImage = 15;

% dataset split percentiles [training, validation, test]
splitPerc = [0.6, 0.1, 0.3];

% yolo input layer size
inputLayerSize = [608, 608, 3];

% dataset table labels: 
className = "Animal";
fileName = "imageFilename";


%% preparing dataset and tables 

% parsing dataset to obtain table in format |image absolute path|xmin, ymin, width, height|
sharkbbdataset = tfDatasetParser("tf_shark_detector_dataset",nImage, fileName, className);

rng("default");
shuffledIndices = randperm(height(sharkbbdataset));
idx = floor(splitPerc(1) * length(shuffledIndices));

trainingIdx = 1:idx;
trainingDataTbl = sharkbbdataset(shuffledIndices(trainingIdx),:);

validationIdx = idx+1 : idx+1+floor(splitPerc(2) * length(shuffledIndices) );
validationDataTbl = sharkbbdataset(shuffledIndices(validationIdx),:);

testIdx = validationIdx(end)+1 : length(shuffledIndices);
testDataTbl = sharkbbdataset(shuffledIndices(testIdx),:);

imdsTrain = imageDatastore(trainingDataTbl{:,fileName});
bldsTrain = boxLabelDatastore(trainingDataTbl(:, className));

imdsValidation = imageDatastore(validationDataTbl{:,fileName});
bldsValidation = boxLabelDatastore(validationDataTbl(:,className));

imdsTest = imageDatastore(testDataTbl{:,fileName});
bldsTest = boxLabelDatastore(testDataTbl(:,className));

trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);

reset(trainingData);

%% creating yolo v4 Object detection network

rng("default")
trainingDataForEstimation = transform(trainingData,@(data)preprocessData(data,inputLayerSize));
numAnchors = 9;
[anchors,meanIoU] = estimateAnchorBoxes(trainingDataForEstimation,numAnchors);

area = anchors(:, 1).*anchors(:,2);
[~,idx] = sort(area,"descend");

anchors = anchors(idx,:);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    anchors(7:9,:)
    };

detector = yolov4ObjectDetector("csp-darknet53-coco",className,anchorBoxes,inputSize=inputLayerSize);


augmentedTrainingData = transform(trainingData,@augmentData);

%training options
options = trainingOptions("adam",...
    GradientDecayFactor=0.9,...
    SquaredGradientDecayFactor=0.999,...
    InitialLearnRate=0.001,...
    LearnRateSchedule="none",...
    MiniBatchSize=4,...
    L2Regularization=0.0005,...
    MaxEpochs=70,...
    BatchNormalizationStatistics="moving",...
    DispatchInBackground=true,...
    ResetInputNormalization=false,...
    Shuffle="every-epoch",...
    VerboseFrequency=20,...
    CheckpointPath=tempdir,...
    ValidationData=validationData);
   
%% Train the YOLO v4 detector.
[detector,info] = trainYOLOv4ObjectDetector(augmentedTrainingData,detector,options);
if ~exist("trained", "dir")
    mkdir("trained");
end
save("trained/", "detector" ...
    )

I = imread("prova.jpg");
[bboxes,scores,labels] = detect(detector,I);

I = insertObjectAnnotation(I,"rectangle",bboxes,scores);
figure
imshow(I)

% test trained detector on a test image
I = imread("test.png");
[bboxes,scores,labels] = detect(detector,I);

I = insertObjectAnnotation(I,"rectangle",bboxes,scores);
figure
imshow(I)

%% evaluate detector on test set

detectionResults = detect(detector,testData);
[ap,recall,precision] = evaluateDetectionPrecision(detectionResults,testData);
figure
plot(recall,precision)
xlabel("Recall")
ylabel("Precision")
grid on
title(sprintf("Average Precision = %.2f",ap))

%% supporting function

function data = preprocessData(data,targetSize)
% Resize the images and scale the pixels to between 0 and 1. Also scale the corresponding bounding boxes.

for ii = 1:size(data,1)
    I = data{ii,1};
    imgSize = size(I);
    
    bboxes = data{ii,2};

    I = im2single(imresize(I,targetSize(1:2)));
    scale = targetSize(1:2)./imgSize(1:2);
    bboxes = bboxresize(bboxes,scale);
    
    data(ii,1:2) = {I,bboxes};
end
end

function data = augmentData(A)
% Apply random horizontal flipping, and random X/Y scaling. Boxes that get
% scaled outside the bounds are clipped if the overlap is above 0.25. Also,
% jitter image color.

data = cell(size(A));
for ii = 1:size(A,1)
    I = A{ii,1};
    bboxes = A{ii,2};
    labels = A{ii,3};
    sz = size(I);

    if numel(sz) == 3 && sz(3) == 3
        I = jitterColorHSV(I,...
            contrast=0.0,...
            Hue=0.1,...
            Saturation=0.2,...
            Brightness=0.2);
    end
    
    % Randomly flip image.
    tform = randomAffine2d(XReflection=true,Scale=[1 1.1]);
    rout = affineOutputView(sz,tform,BoundsStyle="centerOutput");
    I = imwarp(I,tform,OutputView=rout);
    
    % Apply same transform to boxes.
    [bboxes,indices] = bboxwarp(bboxes,tform,rout,OverlapThreshold=0.25);
    labels = labels(indices);
    
    % Return original data only when all boxes are removed by warping.
    if isempty(indices)
        data(ii,:) = A(ii,:);
    else
        data(ii,:) = {I,bboxes,labels};
    end
end
end