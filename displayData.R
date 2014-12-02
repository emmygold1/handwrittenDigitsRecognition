# display data in rows and columns

# set exampleWidth automatically if not passed in
displayData = function(X, exampleWidth)
{
#   if (!exists(exampleWidth) | is.null(exampleWidth))
#   {
#     exampleWidth = round(sqrt(ncol(X)))
#   }
# how to plot gray image in R?
  
# compute rows, cols
  nRows = nrow(X)
  nCols = ncol(X)
  exampleHeight = nCols/exampleWidth
  
# compute number of item to display
  displayRows = floor(sqrt(nRows))
  displayCols = ceiling(nRows/displayRows)
# between images padding

  pad = 1
# setup blank display

  displayArray =  matrix(-1, pad+displayRows*(exampleHeight+pad),
                          pad+displayCols*(exampleWidth+pad))
# copy each example into a patch on the display array
  currEx = 1
  for (j in 1:displayRows)
  {
    for (i in 1:displayCols)
    {
      if (currEx > nRows)
      {
        break
      }
      # copy the patch
      # get the max value of the patch

      displayArray[pad+(j-1)*(exampleHeight+pad)+1:exampleHeight, 
                   pad+(i-1)*(exampleWidth+pad)+1:exampleWidth] = 
        matrix(X[currEx,], exampleHeight, exampleWidth, byrow=T)
# input X must be matrix, otherwise the data.frame arises again after assignment!!!
      currEx = currEx + 1
    }
    if (currEx > nRows)
    {
      break
    }
  }
# display image
# due to the setup in image, data need to be flipped and transposed
  image(t(displayArray[nrow(displayArray):1,]), axes=F, main="Display Data",col=gray((0:32)/32))
}


