using CSV


function dataset(input_file="data/ortho.csv")
    Aᵀ = convert(Array, CSV.read(input_file, header=false))
    A = (Aᵀ)'
end
