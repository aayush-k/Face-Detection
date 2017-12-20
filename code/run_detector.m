% Starter code prepared by James Hays for CS 4476, Georgia Tech
% This function returns detections on all of the images in a given path.
% You will want to use non-maximum suppression on your detections or your
% performance will be poor (the evaluation counts a duplicate detection as
% wrong). The non-maximum suppression is done on a per-image basis. The
% starter code includes a call to a provided non-max suppression function.
% 'test_scn_path' is a string. This directory contains images which may or
%    may not have faces in them. This function should work for the MIT+CMU
%    test set but also for any other images (e.g. class photos)
% 'w' and 'b' are the linear classifier parameters
% 'feature_params' is a struct, with fields
%   feature_params.template_size (default 36), the number of pixels
%      spanned by each train / test template and
%   feature_params.hog_cellSize (default 6), the number of pixels in each
%      HoG cell. template size should be evenly divisible by hog_cellSize.
%      Smaller HoG cell sizes tend to work better, but they make things
%      slower because the feature dimensionality increases and more
%      importantly the step size of the classifier decreases at test time.

% 'bboxes' is Nx4. N is the number of detections. bboxes(i,:) is
%   [x_min, y_min, x_max, y_max] for detection i.
%   Remember 'y' is dimension 1 in Matlab!
% 'confidences' is Nx1. confidences(i) is the real valued confidence of
%   detection i.
% 'image_ids' is an Nx1 cell array. image_ids{i} is the image file name
%   for detection i. (not the full path, just 'albert.jpg')

% The placeholder version of this code will return random bounding boxes in
% each test image. It will even do non-maximum suppression on the random
% bounding boxes to give you an example of how to call the function.

% Your actual code should convert each test image to HoG feature space with
% a _single_ call to vl_hog for each scale. Then step over the HoG cells,
% taking groups of cells that are the same size as your learned template,
% and classifying them. If the classification is above some confidence,
% keep the detection and then pass all the detections for an image to
% non-maximum suppression. For your initial debugging, you can operate only
% at a single scale and you can skip calling non-maximum suppression. Err
% on the side of having a low confidence threshold (even less than zero) to
% achieve high enough recall.
function [bboxes, confidences, image_ids, detectedFeats] = ....
    run_detector(test_scn_path, w, b, feature_params)


    test_scenes = dir( fullfile( test_scn_path, '*.jpg' ));

    % constants
    tempSize = feature_params.template_size;
    cellSize = feature_params.hog_cell_size;
    tempsPerCell = tempSize/cellSize;
    threshold = 1.1; %1.1
    scaleFactor = 0.9; %0.9
    D = tempsPerCell .^ 2 .* 31;


    %initialize these as empty and incrementally expand them.
    bboxes = zeros(0,4);
    confidences = zeros(0,1);
    image_ids = cell(0,1);
    detectedFeats = [];


    for i = 1 : length(test_scenes)

        fprintf('Detecting faces in %s\n', test_scenes(i).name)
        % initialize bbox, confs, imageIds, and scale for current scene
        sceneBBoxes = zeros(0, 4);
        sceneConfs = zeros(0, 1);
        sceneImageIds = cell(0, 1);
        sceneFeats = [];

        sceneZoom = 1;

         % open and grayscale image
        img = imread( fullfile( test_scn_path, test_scenes(i).name ));
        inputImSize = size(img);
        img = single(img)/255;
        if(size(img,3) > 1)
            img = rgb2gray(img);
        end

        iter = 0;
        while (size(img, 1) > tempSize || size(img, 2) > tempSize)
            % if strcmp(test_scenes(i).name, '251966.jpg')
            %     fprintf('\titeration # %d at zoom %d\n', iter, sceneZoom);
            % end
            % get current image's hog
            imHog = vl_hog(img, cellSize);

            % sliding window scan on overall hog
            for r = 1 : (size(imHog, 1) - tempsPerCell)
                for c = 1 : (size(imHog, 2) - tempsPerCell)
                    % get current sliding window, excluding last row/col for correct dimensionality
                    currHogWindow = reshape(imHog(r : (r + tempsPerCell - 1), c : (c + tempsPerCell - 1), :), 1, D, 1);
                    conf = currHogWindow * w + b;

                        % successfully confident detection results in adding it to our output
                    if (conf >= threshold)
                        % APPLY ANY ADDITIONAL CASCADE CLASSIFIERS HERE
                        % get top left and bottom Right pixel in the current scaled dimensions
                        topLeft = ([c r] - 1) .* cellSize;
                        bottomRight = topLeft + tempSize;
                         % gets original pixel coordinates
                        hogWindow = round([topLeft bottomRight] .* sceneZoom);
                        % add detection to output
                        sceneBBoxes = [sceneBBoxes; hogWindow];
                        sceneConfs = [sceneConfs; conf];
                        sceneImageIds = [sceneImageIds; test_scenes(i).name];
                        sceneFeats = [sceneFeats; currHogWindow];
                    end
                end
            end
            % scale image again to run detection window on next gaussian pyramid
            img = imresize(img, scaleFactor);
            sceneZoom = sceneZoom / scaleFactor;
            iter = iter + 1;
        end


        %non_max_supr_bbox can actually get somewhat slow with thousands of
        %initial detections. You could pre-filter the detections by confidence,
        %e.g. a detection with confidence -1.1 will probably never be
        %meaningful. You probably _don't_ want to threshold at 0.0, though. You
        %can get higher recall with a lower threshold. You don't need to modify
        %anything in non_max_supr_bbox, but you can.
        [is_maximum] = non_max_supr_bbox(sceneBBoxes, sceneConfs, inputImSize);
        % do non max filtering
        sceneConfs = sceneConfs(is_maximum,:);
        sceneBBoxes      = sceneBBoxes(is_maximum,:);
        sceneImageIds   = sceneImageIds(is_maximum,:);
        sceneFeats   = sceneFeats(is_maximum,:);
        % append to running return value matrices
        bboxes = [bboxes; sceneBBoxes];
        confidences = [confidences; sceneConfs];
        image_ids = [image_ids; sceneImageIds];
        detectedFeats = [detectedFeats; sceneFeats];
    end
end




