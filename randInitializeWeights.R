randInitializeWeights = function (lIn, lOut)
{
  epsilonInit = sqrt(6/(lIn + lOut))
  W = matrix(runif((lIn+1)*lOut, -epsilonInit, epsilonInit), lOut, lIn+1)
  return(W)
}