% Starter code prepared by James Hays for CS 4476, Georgia Tech
% This function should return all positive training examples (faces) from
% 36x36 images in 'train_path_pos'. Each face should be converted into a
% HoG template according to 'feature_params'. For improved performance, try
% mirroring or warping the positive training examples to augment your
% training data.

function features_pos = get_positive_features(train_path_pos, feature_params)
    % 'train_path_pos' is a string. This directory contains 36x36 images of
    %   faces
    % 'feature_params' is a struct, with fields
    %   feature_params.template_size (default 36), the number of pixels
    %      spanned by each train / test template and
    %   feature_params.hog_cell_size (default 6), the number of pixels in each
    %      HoG cell. template size should be evenly divisible by hog_cell_size.
    %      Smaller HoG cell sizes tend to work better, but they make things
    %      slower because the feature dimensionality increases and more
    %      importantly the step size of the classifier decreases at test time.
    %      (although you don't have to make the detector step size equal a
    %      single HoG cell).


    % 'features_pos' is N by D matrix where N is the number of faces and D
    % is the template dimensionality, which would be
    %   (feature_params.template_size / feature_params.hog_cell_size)^2 * 31
    % if you're using the default vl_hog parameters

    % Useful functions:
    % single(image)
    % reshape here but not in posititve
    % vl_hog, HOG = VL_HOG(IM, CELLSIZE)
    %  http://www.vlfeat.org/matlab/vl_hog.html  (API)
    %  http://www.vlfeat.org/overview/hog.html   (Tutorial)
    % rgb2gray

    image_files = dir( fullfile( train_path_pos, '*.jpg') ); %Caltech Faces stored as .jpg
    num_images = length(image_files);


    D = (feature_params.template_size / feature_params.hog_cell_size)^2 * 31;
    features_pos = zeros(num_images * 2, D);

    counter = 1;
    for i = 1:num_images
        % if i >= 6700 || mod(i, 100) == 0
        %     fprintf('pos processed %d of %d images\n', i, num_images);
        % end
        imFile = strcat(train_path_pos, '/', image_files(i).name);
        img = single(imread(imFile)); % consider HSV?


        hog = vl_hog(img, feature_params.hog_cell_size);
        features_pos(counter,:) = reshape(hog, D, 1, 1);

        counter = counter + 1;
        img = flip(img, 2);

        hog = vl_hog(img, feature_params.hog_cell_size);
        features_pos(counter,:) = reshape(hog, D, 1, 1);
    end
    fprintf('done getting pos feats\n');
end