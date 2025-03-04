function constrain!(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    cons::Vector{<:AbstractLinearConstraint};
    verbose=false
)
    for con! ∈ cons
        if verbose
            println("        applying constraint: ", con!.label)
        end
        con!(opt, vars)
    end
end

function (con::EqualityConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex}
)
    for (i, val) ∈ zip(con.indices, con.values)
        MOI.add_constraints(
            opt,
            vars[i],
            MOI.EqualTo(val)
        )
    end
end

function (con::BoundsConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex}
)
    for (i, (lb, ub)) ∈ zip(con.indices, con.bounds)
        MOI.add_constraints(
            opt,
            vars[i],
            MOI.GreaterThan(lb)
        )
        MOI.add_constraints(
            opt,
            vars[i],
            MOI.LessThan(ub)
        )
    end
end

function (con::AllEqualConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex}
)
    x_minus_val = MOI.ScalarAffineTerm(-1.0, vars[con.bar_index])
    for i ∈ con.indices
        xᵢ = MOI.ScalarAffineTerm(1.0, vars[i])
        MOI.add_constraints(
            opt,
            MOI.ScalarAffineFunction([xᵢ, x_minus_val], 0.0),
            MOI.EqualTo(0.0)
        )
    end
end

function (con::L1SlackConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex}
)
    for (x, s1, s2) in zip(con.x_indices, con.s1_indices, con.s2_indices)
        MOI.add_constraints(
            opt,
            vars[s1],
            MOI.GreaterThan(0.0)
        )
        MOI.add_constraints(
            opt,
            vars[s2],
            MOI.GreaterThan(0.0)
        )
        t1 = MOI.ScalarAffineTerm(1.0, vars[s1])
        t2 = MOI.ScalarAffineTerm(-1.0, vars[s2])
        t3 = MOI.ScalarAffineTerm(-1.0, vars[x])
        MOI.add_constraints(
            opt,
            MOI.ScalarAffineFunction([t1, t2, t3], 0.0),
            MOI.EqualTo(0.0)
        )
    end
end