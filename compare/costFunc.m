function [cost grad] = costFunc(v)
  cost = sum(v.^4);
  grad = 4*v.^3;
end

