export KetIntegrator
export UnitaryIntegrator
export DensityMatrixIntegrator

const ⊗ = kron

function KetIntegrator(
    sys::QuantumSystem,
    traj::NamedTrajectory, 
    ψ̃::Symbol, 
    a::Symbol, 
    Δt::Symbol
) 
    return BilinearIntegrator(sys.G, traj, ψ̃, a, Δt)
end

function UnitaryIntegrator(
    sys::QuantumSystem,
    traj::NamedTrajectory, 
    Ũ⃗::Symbol, 
    a::Symbol, 
    Δt::Symbol
) 
    Ĝ = a_ -> I(sys.levels) ⊗ sys.G(a_)
    return BilinearIntegrator(Ĝ, traj, Ũ⃗, a, Δt)
end

function DensityMatrixIntegrator(
    sys::OpenQuantumSystem,
    traj::NamedTrajectory, 
    ρ̃::Symbol, 
    a::Symbol, 
    Δt::Symbol
) 
    return BilinearIntegrator(sys.𝒢, traj, ρ̃, a, Δt)
end