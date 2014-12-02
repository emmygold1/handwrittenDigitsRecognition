computeNumericalGradient = function (J, theta)
{
  numgrad = rep(0, length(theta))
  perturb = numgrad
  e = 1e-4
  for (p in 1:length(theta))
  {
    perturb[p] = e
    loss1 = J(theta - perturb)[1]
    loss2 = J(theta + perturb)[1]
    numgrad[p] = (loss2 - loss1)/2/e
    perturb[p] = 0
  }
  return(numgrad)
}