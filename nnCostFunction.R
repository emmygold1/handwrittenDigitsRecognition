nnCostFunction = function (nnParams, inputLayerSize, hiddenLayerSize,
                           numLabels, X, y, lambda)
{
  # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
  # for our 2 layer neural network
  theta1 = matrix(nnParams[1:(hiddenLayerSize*(inputLayerSize + 1))],
                  hiddenLayerSize, (inputLayerSize+1))
  theta2 = matrix(nnParams[(1+hiddenLayerSize*(inputLayerSize+1)):length(nnParams)],
                  numLabels, (hiddenLayerSize+1))
# Setup some useful variables
  m = nrow(X)
  J = 0
  theta1Grad = matrix(0, nrow(theta1), ncol(theta1))
  theta2Grad = matrix(0, nrow(theta2), ncol(theta2))
  a1 = cbind(rep(1, m), X)
  z2 = a1 %*% t(theta1)
  a2 = sigmoid(z2)
  a2 = cbind(rep(1,m), a2)
  z3 = a2 %*% t(theta2)
  a3 = sigmoid(z3)
# transform y value into vectors
  yMat = matrix(0, m, numLabels)
# transform the mth value of y into a vector with the y'th value being one
  ind = (y-1)*m + 1:m
  yMat[ind] = 1

  J = -sum(yMat*log(a3) + (1-yMat)*log(1-a3))/m + 
      sum(theta1[,2:ncol(theta1)]^2)*lambda/2/m +
      sum(theta2[,2:ncol(theta2)]^2)*lambda/2/m

  d3 = a3 - yMat
  d3Theta2 = d3 %*% theta2
  d2 = d3Theta2[,2:ncol(d3Theta2)] * sigmoidGradient(z2)

  theta1Grad = t(d2) %*% a1/m
  theta2Grad = t(d3) %*% a2/m

  theta1Grad[,2:ncol(theta1Grad)] = theta1Grad[,2:ncol(theta1Grad)] + 
                                   lambda/m*theta1[,2:ncol(theta1)]
  theta2Grad[,2:ncol(theta2Grad)] = theta2Grad[,2:ncol(theta2Grad)] + 
                                   lambda/m*theta2[,2:ncol(theta2)]
  grad = c(as.vector(theta1Grad), as.vector(theta2Grad))
  attr(J, "gradient") = grad # gradient attribute for minimisation
  return(J)
}