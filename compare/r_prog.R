source("costFunc.R")
source("grad.R")
startTime = proc.time()
results = optimx(1:1000000, fn=costFunc, gr=grad,
                 itnmax=50, method=c("Rcgmin"))
proc.time() - startTime