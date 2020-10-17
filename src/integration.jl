module AttractorNets
module Integration

summation(W, a, i) = W[1:end.!==i,i]' * a[1:end.!==i]
export summation

end # Integration
end # AttractorNets
