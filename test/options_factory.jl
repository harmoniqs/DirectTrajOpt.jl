module ParameterFactory

abstract type absty end
struct conty1 <: absty
    k::Symbol
    v::Int
    c::String
end
struct conty2 <: absty
    k::Symbol
    v::Float64
    c::String
end

function conty1()
    "hello"
end
function conty2()
    "bye"
end

const abstyrefty = Ref{Type{<:absty}}
const defaultty::abstyrefty = Ref{Type{<:absty}}(conty1)

##

function myfunc(x; params = ())
    return
end

# struct 

end
