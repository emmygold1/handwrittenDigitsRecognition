# handwritten digits recognition
rm(list=ls())
graphics.off()
setwd("~/Documents/kaggle/handWrittenDigits")
source("checkNnGradients.R")
source("computeNumericalGradient.R")
source("debugInitializeWeights.R")
source("displayData.R")
source("nnCostFunction.R")
source("predict.R")
source("randInitializeWeights.R")
source("sigmoid.R")
source("sigmoidGradient.R")
dat = read.csv("train.csv", header=T)
# transform data frame into matrix
# otherwise subtle problems pop up
dat = as.matrix(dat)
nRow = nrow(dat)
nCol = ncol(dat)
y = dat[,1] + 1 # conform to array index starting from 1
x = dat[,2:nCol]
pixelSize = 28
inputLayerSize = pixelSize^2
hiddenLayerSize = 3
numLabels = 10
# have a quick look of 100 randomly selected examples
dis = sample(nRow, 100)
yDis = y[dis]
xDis = x[dis,]
displayData(xDis,28)
# initialize the thetas
theta1 = randInitializeWeights(inputLayerSize, hiddenLayerSize)
theta2 = randInitializeWeights(hiddenLayerSize, numLabels)
initialNnParams = c(as.vector(theta1), as.vector(theta2))
lambda = .1
costFunction = function(p) nnCostFunction(p, inputLayerSize, hiddenLayerSize,
                                          numLabels, x[1:2,], y[1:2], lambda) 
results = nlm(costFunction, initialNnParams,iterlim=1, print.level=0)
results$minimum

theta1 = matrix(results$estimate[1:(hiddenLayerSize*(inputLayerSize+1))],
                hiddenLayerSize, inputLayerSize+1)
theta2 = matrix(results$estimate[(1+hiddenLayerSize*(inputLayerSize+1)):
                                   length(results$estimate)], numLabels,
                (hiddenLayerSize+1))

pred = predict(theta1, theta2, x) - 1
y = y - 1
eIn = length(which(pred != y))/nRow



