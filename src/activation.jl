module AttractorNets
module Activation

sign_activation(inᵢ, a₀)::Int = inᵢ==0 ? a₀ : Int(sign(inᵢ))
export sign_activation

end # Integration
end # AttractorNets
