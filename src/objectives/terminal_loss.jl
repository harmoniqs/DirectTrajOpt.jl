export TerminalLoss

function TerminalLoss(
    ℓ::Function,
    name::Symbol,
    traj::NamedTrajectory;
    Q::Float64=1.0
)
    return KnotPointObjective(
        ℓ,
        name,
        traj;
        Qs=[Q],
        times=[traj.T]
    )
end
