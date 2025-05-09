export GlobalObjective
export GlobalKnotPointObjective


# ----------------------------------------------------------------------------- #
# GlobalObjective
# ----------------------------------------------------------------------------- #

"""
    GlobalObjective(
        ℓ::Function,
        global_names::AbstractVector{Symbol},
        traj::NamedTrajectory;
        kwargs...
    )
    GlobalObjective(
        ℓ::Function,
        global_name::Symbol,
        traj::NamedTrajectory;
        kwargs...
    )

Create an objective that only involves the global components.
"""
function GlobalObjective(
    ℓ::Function,
    global_names::AbstractVector{Symbol},
    traj::NamedTrajectory;
    Q::Float64=1.0
)
    Z_dim = traj.dim * traj.T + traj.global_dim
    g_comps = vcat([traj.global_components[name] for name in global_names]...)
    
    L(Z⃗::AbstractVector{<:Real}) = Q * ℓ(Z⃗[g_comps])

    @views function ∇L(Z⃗::AbstractVector{<:Real})
        ∇ = zeros(Z_dim)
        ∇[g_comps] = ForwardDiff.gradient(x -> Q * ℓ(x), Z⃗[g_comps])
        return ∇
    end

    function ∂²L_structure()
        structure = spzeros(Z_dim, Z_dim)
        structure[g_comps, g_comps] .= 1.0
        structure_pairs = collect(zip(findnz(structure)[1:2]...))
        return structure_pairs
    end

    @views function ∂²L(Z⃗::AbstractVector{<:Real})
        ∂²ℓ = ForwardDiff.hessian(x -> Q * ℓ(x), Z⃗[g_comps])
        return ∂²ℓ[:]
    end

    return Objective(L, ∇L, ∂²L, ∂²L_structure)
end

function ℓ(ℓ::Function, global_name::Symbol, traj::NamedTrajectory; kwargs...)
    return GlobalObjective(ℓ, [global_name], traj; kwargs...)
end

# ----------------------------------------------------------------------------- #
# Global KnotPointObjective
# ----------------------------------------------------------------------------- #

function KnotPointObjective(
    ℓ::Function,
    names::AbstractVector{Symbol},
    global_names::AbstractVector{Symbol},
    traj::NamedTrajectory,
    params::AbstractVector;
    times::AbstractVector{Int}=1:traj.T,
    Qs::AbstractVector{Float64}=ones(traj.T),
)
    @assert length(Qs) == length(times) "Qs must have the same length as times"
    @assert length(params) == length(times) "params must have the same length as times"

    Z_dim = traj.dim * traj.T + traj.global_dim
    x_comps = vcat([traj.components[name] for name in names]...)
    g_comps = vcat([traj.global_components[name] for name in global_names]...)
    xg_slices = [vcat([slice(t, x_comps, traj.dim), g_comps]...) for t in times]
    
    function L(Z⃗::AbstractVector{<:Real})
        loss = 0.0
        for (i, x_slice) in enumerate(xg_slices)
            x = Z⃗[x_slice]
            loss += Qs[i] * ℓ(x, params[i])
        end
        return loss
    end

    @views function ∇L(Z⃗::AbstractVector{<:Real})
        ∇ = zeros(Z_dim)
        for (i, x_slice) in enumerate(xg_slices)
            # Add because global params
            ∇[x_slice] .+= ForwardDiff.gradient(x -> Qs[i] * ℓ(x, params[i]), Z⃗[x_slice])
        end
        return ∇
    end

    function ∂²L_structure()
        structure = spzeros(Z_dim, Z_dim)
        for x_slice in xg_slices
            structure[x_slice, x_slice] .= 1.0
        end
        structure_pairs = collect(zip(findnz(structure)[1:2]...))
        return structure_pairs
    end

    function ∂²L_structure_mapping()
        # Build a mapping from (i, j) -> idx
        structure_dict = Dict{Tuple{Int, Int}, Int}()
        for (idx, pair) in enumerate(∂²L_structure())
            structure_dict[pair] = idx
        end
        return structure_dict
    end

    @views function ∂²L(Z⃗::AbstractVector{<:Real})
        structure_dict = ∂²L_structure_mapping()
        ∂²L_values = zeros(length(structure_dict))
        for (i, x_slice) in enumerate(xg_slices)
            ∂²ℓ = ForwardDiff.hessian(x -> Qs[i] * ℓ(x, params[i]), Z⃗[x_slice])

            # TODO: Is there a more efficient way to do this?
            for local_i in eachindex(x_slice)
                global_i = x_slice[local_i]
                for local_j in 1:local_i
                    global_j = x_slice[local_j]
    
                    # Add to (i,j)
                    idx = structure_dict[(global_i, global_j)]
                    ∂²L_values[idx] += ∂²ℓ[local_i, local_j]
    
                    # Add to (j,i) if off-diagonal
                    if local_i != local_j
                        idx_sym = structure_dict[(global_j, global_i)]
                        ∂²L_values[idx_sym] += ∂²ℓ[local_j, local_i]
                    end
                end
            end
        end
        return ∂²L_values
    end

    return Objective(L, ∇L, ∂²L, ∂²L_structure)
end

function KnotPointObjective(
    ℓ::Function,
    names::AbstractVector{Symbol},
    global_names::AbstractVector{Symbol},
    traj::NamedTrajectory;
    times::AbstractVector{Int}=1:traj.T,
    kwargs...
)
    params = [nothing for _ in times]
    ℓ_param = (x, _) -> ℓ(x)
    return KnotPointObjective(ℓ_param, names, global_names, traj, params; times=times, kwargs...)
end


function TerminalObjective(
    ℓ::Function,
    name::Symbol,
    global_names::AbstractVector{Symbol},
    traj::NamedTrajectory;
    Q::Float64=1.0,
    kwargs...
)
    return KnotPointObjective(
        ℓ,
        name,
        global_names,
        traj;
        Qs=[Q],
        times=[traj.T],
        kwargs...
    )
end