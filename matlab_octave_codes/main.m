%% Machine Learning for Kaggle handwritten digits recognisor competition
%% use mini-batch


%% Initialization
clear ; close all; clc

%% Setup the parameters you will use for this exercise
input_layer_size  = 784;  % 28x28 Input Images of Digits
hidden_layer_size = 250;   % 250 hidden units
num_labels = 10;          % 10 labels, from 1 to 10   
                          % (note that we have mapped "0" to label 10)

%% =========== Part 1: Loading and Visualizing Data =============
%  We start by first loading and visualizing the dataset. 
%  You will be working with a dataset that contains handwritten digits.
%

% Load Training Data
fprintf('Loading Data ...\n')

dat = csvread('train.csv'); % be careful csvread read the head as all zeros
dat = dat(2:end,:);
no_train = size(dat, 1);
perm = randperm(no_train);
X = dat(:, 2:end);
y = dat(:,1) + 1; % adapted to 1-based array index

X_cv = dat(perm(41001:end), 2:end);
y_cv = dat(perm(41001:end), 1) + 1;
clear dat; %remove dat variable to save more memory



% Randomly select 100 data points to display
%  sel = randperm(size(X, 1));
%  sel = sel(1:100);
%  
%  displayData(X(sel, :));
%  
%  fprintf('Program paused. Press enter to continue.\n');
%  pause;

% form mini-batch

mini_batch_inits = 1:10:no_train;
mini_batchs_length = length(mini_batch_inits);
cost_train = zeros(mini_batchs_length,1);
cost_cv = zeros(mini_batchs_length,1);

%% ================ Part 2: Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
lambda = 20;
options = optimset('MaxIter', 200);

%% =================== Part 3: Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%

for iter_mini_batch = 1:mini_batchs_length
  mini_batch_init = mini_batch_inits(iter_mini_batch);
  fprintf('\nTraining Neural Network... %dth mini-batch\n', iter_mini_batch)
% Create "short hand" for the cost function to be minimized

costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X(perm(mini_batch_init:mini_batch_init+9),:),...
                                   y(perm(mini_batch_init:mini_batch_init+9)), lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
tic
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);
toc

initial_nn_params = nn_params;
cost_train(iter_mini_batch) = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X,...
                                   y, 0);
cost_cv(iter_mini_batch) = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X_cv,...
                                   y_cv, 0);
end
save costs.m cost_train cost_cv;
% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

save thetas.m Theta1 Theta2;

%% ================= Part 4: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

test_dat = csvread('test.csv'); % be careful csvread read the head as all zeros
test_dat = test_dat(2:end,:);

pred = predict(Theta1, Theta2, test_dat) - 1;
if size(pred,1) == 1
  save test_pred.dat pred' -ascii;
else
  save test_pred.dat pred -ascii;
end
% only save a column

%  %% ================= Part 5: Visualize Weights =================
%  %  You can now "visualize" what the neural network is learning by 
%  %  displaying the hidden units to see what features they are capturing in 
%  %  the data.
%  
%  fprintf('\nVisualizing Neural Network... \n')
%  
%  displayData(Theta1(:, 2:end));
%  
%  fprintf('\nProgram paused. Press enter to continue.\n');
%  pause;
