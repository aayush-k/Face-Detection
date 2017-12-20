% Starter code prepared by James Hays for CS 4476, Georgia Tech
% This function should return negative training examples (non-faces) from
% any images in 'non_face_scn_path'. Images should be converted to
% grayscale because the positive training data is only available in
% grayscale.

function features_neg = get_random_negative_features(non_face_scn_path, feature_params, num_samples)
    % 'non_face_scn_path' is a string. This directory contains many images
    %   which have no faces in them.
    % 'feature_params' is a struct, with fields
    %   feature_params.template_size (default 36), the number of pixels
    %      spanned by each train / test template and
    %   feature_params.hog_cell_size (default 6), the number of pixels in each
    %      HoG cell. template size should be evenly divisible by hog_cell_size.
    %      Smaller HoG cell sizes tend to work better, but they make things
    %      slower because the feature dimensionality increases and more
    %      importantly the step size of the classifier decreases at test time.
    % 'num_samples' is the number of random negatives to be mined, it's not
    %   important for the function to find exactly 'num_samples' non-face
    %   features, e.g. you might try to sample some number from each image, but
    %   some images might be too small to find enough.

    % 'features_neg' is N by D matrix where N is the number of non-faces and D
    % is the template dimensionality, which would be
    %   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
    % if you're using the default vl_hog parameters

    % Useful functions:
    % vl_hog, HOG = VL_HOG(IM, CELLSIZE)
    %  http://www.vlfeat.org/matlab/vl_hog.html  (API)
    %  http://www.vlfeat.org/overview/hog.html   (Tutorial)
    % rgb2gray

    image_files = dir( fullfile( non_face_scn_path, '*.jpg' ));
    num_images = length(image_files);
    cellSize = feature_params.hog_cell_size;
    tempSize = feature_params.template_size;
    D = (tempSize / cellSize)^2 * 31;

    standardSampleSize = ceil(num_samples / num_images);
    features_neg = zeros(num_images, D); %number of samples?
    featIndex = 1;

    % 0.7 reccomended by Dr.Hays for tradeoff between accuracy and performance
    scaleFactor = 0.7;

    for i = 1:num_images
        if mod(i, 20) == 0
            fprintf('neg processed %d of %d images\n', i, num_images);
        end

        imFile = strcat(non_face_scn_path, '/', image_files(i).name);
        % fprintf('getting random features in %s\n', imFile);
        img = single(rgb2gray(imread(imFile))); % consider HSV?
        [imHeight, imWidth] = size(img);
        % For best performance, you should sample random negative examples at multiple scales.

        % how many nonoverlapping templates can fit in the current image
        uniqueFeatures = floor(imHeight / tempSize) * floor(imWidth / tempSize);
        featuresSampled = min([round(standardSampleSize / 2), uniqueFeatures]);

        % while currently scaled image can contain at least one nonoverlapping image
        while (imHeight >= tempSize && imWidth >= tempSize)

            for fS = 1 : featuresSampled
                % top left coordinates of feature
                r = ceil(rand * (imHeight - tempSize) + 1);
                c = ceil(rand * (imWidth - tempSize) + 1);
                imWindow = img(r : r + tempSize - 1, c : c + tempSize - 1);
                %linearize into row vec
                feature = reshape(vl_hog(imWindow, cellSize), 1, []);
                features_neg(featIndex,:) = feature; % append row vec
                featIndex = featIndex + 1;
            end
            img = imresize(img, scaleFactor);
            [imHeight, imWidth] = size(img);
            % num nonoverlapping templates that fit in the resized image
            uniqueFeatures = floor(imHeight / tempSize) * floor(imWidth / tempSize);
            featuresSampled = min([featuresSampled * scaleFactor .^ 2, uniqueFeatures]);
            % if there are no features left to sample because image is scaled too small, then break
            if featuresSampled <= 0
                break
            end
        end

    end
    % filter down to only required number of samples
    % TODO: implement more even distribution of sampling across ranges of image zoom scales
    if (featIndex - 1 > num_samples)
        features_neg = datasample(features_neg, num_samples, 'Replace', false);
    end
    fprintf('done getting neg feats\n');
end
