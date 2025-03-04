export TerminalLoss

function TerminalLoss(
    ℓ::Function,
    name::Symbol,
    traj::NamedTrajectory;
    Q::Float64=1.0
)
    Z_dim = traj.dim * traj.T + traj.global_dim
    x_T_slice = slice(traj.T, traj.components[name], traj.dim)

    function L(Z⃗::AbstractVector)
        x_T = Z⃗[x_T_slice]
        return Q * ℓ(x_T)
    end

    @views function ∇L(Z⃗::AbstractVector)
        ∇ = zeros(Z_dim)
        ForwardDiff.gradient!(∇[x_T_slice], x -> Q * ℓ(x), Z⃗[x_T_slice]) 
        return ∇
    end

    function ∂²L_structure()
        structure = spzeros(Z_dim, Z_dim)
        structure[x_T_slice, x_T_slice] .= 1.0
        structure_pairs = collect(zip(findnz(structure)[1:2]...))
        return structure_pairs
    end

    function ∂²L(Z⃗::AbstractVector)
        ∂²ℓ = ForwardDiff.hessian(x -> Q * ℓ(x), Z⃗[x_T_slice])
        return ∂²ℓ[:]
    end

    return Objective(L, ∇L, ∂²L, ∂²L_structure)
end



