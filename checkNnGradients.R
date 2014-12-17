checkNnGradients = function (lambda)
{
  inputLayerSize = 3
  hiddenLayerSize = 5
  numLabels = 3
  m = 5
  theta1 = debugInitializeWeights(hiddenLayerSize, inputLayerSize)
  theta2 = debugInitializeWeights(numLabels, hiddenLayerSize)
  
  x = debugInitializeWeights(m, inputLayerSize - 1)
  y = 1 + (1:m) %% numLabels
  nnParams = c(as.vector(theta1), as.vector(theta2))
  costFunc = function(p) nnCostFunction(p, inputLayerSize,
                                     hiddenLayerSize, numLabels, x, y, lambda)
  grad = nnGradientFunction(nnParams, inputLayerSize,
                            hiddenLayerSize, numLabels, x, y, lambda)
  numgrad = computeNumericalGradient(costFunc, nnParams)
  print(cbind(numgrad, grad))
  writeLines(c("The above two columns you get should be very similar",
               "(Left-Your Numerical Gradient, Right-Analytical Gradient)"))
  diff = norm(as.matrix(numgrad - grad),"f")/norm(as.matrix(numgrad + grad),"f")
  cat(sprintf(c("If your backpropagation implementation is correct, then\n",
                "the relative difference will be small (less than 1e-9). \n",
                "\nRelative Difference: %g\n"), diff))
}
