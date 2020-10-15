using Serialization

named_numbers = ["one","two","three","four","five","six","seven","eight","nine","zero"]

function dataset(numbers)
    indices = replace(x->x==0 ? 10 : x, numbers)
    names = named_numbers[indices]

    A = hcat([reshape(deserialize("data/$(name).dat"),64) for name in names]...)
end
