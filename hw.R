# handwritten digits recognition
# don't forget to load library optimx!!!!!
rm(list=ls())
graphics.off()
setwd("~/Documents/kaggle/handWrittenDigits")
source("checkNnGradients.R")
source("computeNumericalGradient.R")
source("debugInitializeWeights.R")
source("displayData.R")
source("nnCostFunction.R")
source("nnGradientFunction.R")
source("predict.R")
source("randInitializeWeights.R")
source("sigmoid.R")
source("sigmoidGradient.R")
dat = read.csv("train.csv", header=T)
# transform data frame into matrix
# otherwise subtle problems pop up
dat = as.matrix(dat)
nTrains = nrow(dat)
nFeatures = ncol(dat) - 1
y = dat[,1] + 1 # conform to 1 based array
x = dat[,2:(nFeatures+1)]
pixelSize = 28
inputLayerSize = pixelSize^2
if (inputLayerSize != nFeatures) {stop("Data features unequal to input size.")}
hiddenLayerSize = 5
numLabels = 10
cat(sprintf("You will train a neural network with"))
#------------------------------------------------------
# have a quick look of 100 randomly selected examples
# dis = sample(nRow, 100)
# yDis = y[dis]
# xDis = x[dis,]
# displayData(xDis,28)
#------------------------------------------------------
# initialize the thetas
theta1 = randInitializeWeights(inputLayerSize, hiddenLayerSize)
theta2 = randInitializeWeights(hiddenLayerSize, numLabels)
initialNnParams = c(as.vector(theta1), as.vector(theta2))
lengthIniNnPa = length(initialNnParams)
lambda = 1
# costFunction = function(p) nnCostFunction(p, inputLayerSize, hiddenLayerSize,
#                                           numLabels, x[1:2,], y[1:2], lambda) 
# gradientFunction = function(p) nnGradientFunction(p, inputLayerSize, hiddenLayerSize,
#                                           numLabels, x[1:2,], y[1:2], lambda)
startTime = proc.time()
results = optimx(initialNnParams, fn=nnCostFunction, gr=nnGradientFunction,
                 lower=-10, upper=10,
                 itnmax=5, method=c("Nelder-Mead", "BFGS"), inputLayerSize=inputLayerSize, 
                 hiddenLayerSize=hiddenLayerSize,
                 numLabels=numLabels, X=x[1:2,], y=y[1:2], lambda=lambda)
proc.time() - startTime

results = as.matrix(results[1,1:lengthIniNnPa])

theta1 = matrix(results[,1:(hiddenLayerSize*(inputLayerSize+1))],
                hiddenLayerSize, inputLayerSize+1)
theta2 = matrix(results[,(1+hiddenLayerSize*(inputLayerSize+1)):
                          lengthIniNnPa], numLabels,
                (hiddenLayerSize+1))

pred = predict(theta1, theta2, x) - 1
y = y - 1
eIn = length(which(pred != y))/nTrains



