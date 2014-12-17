predict = function(theta1, theta2, X)
{
  nRowX = nrow(X)
  h1 = sigmoid(cbind(rep(1,nRowX), X) %*% t(theta1))
  h2 = sigmoid(cbind(rep(1,nRowX), h1) %*% t(theta2))
  return(apply(h2, 1, which.max))
}