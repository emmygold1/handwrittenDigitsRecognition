debugInitializeWeights = function (fanOut, fanIn)
{
  w = matrix(sin(1:(fanOut*(1+fanIn))), fanOut, 1+fanIn)/10
  return(w)
}