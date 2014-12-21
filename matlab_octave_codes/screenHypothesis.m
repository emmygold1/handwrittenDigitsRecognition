%% Machine Learning for Kaggle handwritten digits recognisor competition


%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits

num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading Data ...\n')

dat = csvread('train.csv');
dat = dat(2:end,:);
m = size(dat, 1);
perm = randperm(m,1000);
X = dat(perm(1:700), 2:end);
y = dat(perm(1:700),1) + 1; % adapted to 1-based array index

X_cv = dat(perm(701:1000), 2:end);
y_cv = dat(perm(701:1000),1) + 1; % adapted to 1-based array index
hidden_layer_sizes = 50:20:510;
hidden_layer_sizes_length = length(hidden_layer_sizes);
cost_train = zeros(hidden_layer_sizes_length, 1);
cost_cv = zeros(hidden_layer_sizes_length, 1);
% Randomly select 100 data points to display
%  sel = randperm(size(X, 1));
%  sel = sel(1:100);
%  
%  displayData(X(sel, :));
%  
%  fprintf('Program paused. Press enter to continue.\n');
%  pause;

%% ================ Part 2: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

for iter_screen =  1:hidden_layer_sizes_length  % screen hidden layer size
  hidden_layer_size = hidden_layer_sizes(iter_screen);
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];



%% =================== Part 3: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network...%d \n', iter_screen)

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
options = optimset('MaxIter', 100);

%  You should also try different values of lambda
lambda = 0;

% Create "short hand" for the cost function to be minimized

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
tic
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
toc
cost_cv(iter_screen) = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_cv, y_cv, lambda);
cost_train(iter_screen) = cost(end);
end % end screening hidden_layer_size
figure
hold on
plot(hidden_layer_sizes, cost_cv,'.-')
plot(hidden_layer_sizes, cost_train, 'r.-')
hold off

% Obtain Theta1 and Theta2 back from nn_params
%  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
%                   hidden_layer_size, (input_layer_size + 1));
%  
%  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
%                   num_labels, (hidden_layer_size + 1));
%  
%  fprintf('Program paused. Press enter to continue.\n');
%  pause;

%% ================= Part 4: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

%  pred = predict(Theta1, Theta2, X);
%  
%  fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

%% ================= Part 5: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.
%  
%  fprintf('\nVisualizing Neural Network... \n')
%  
%  displayData(Theta1(:, 2:end));
%  
%  fprintf('\nProgram paused. Press enter to continue.\n');
%  pause;
