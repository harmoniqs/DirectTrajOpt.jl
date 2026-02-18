function constrain!(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    cons::Vector{<:AbstractLinearConstraint},
    traj::NamedTrajectory;
    verbose=false
)
    for con! ∈ cons
        if verbose
            println("        applying constraint: ", con!.label)
        end
        # Apply constraint to optimizer (computes indices internally)
        con!(opt, vars, traj)
    end
    return nothing
end

function (con::EqualityConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
)
    name = con.var_names
    
    if con.is_global
        # Global variable constraint
        @assert name ∈ traj.global_names "Global variable $name not found in trajectory"
        @assert length(con.values) == traj.global_dims[name] "Value dimension mismatch for global variable $name"
        
        indices = traj.dim * traj.N .+ traj.global_components[name]
        
        for (i, val) ∈ zip(indices, con.values)
            MOI.add_constraints(opt, vars[i], MOI.EqualTo(val))
        end
    else
        # Trajectory variable constraint
        @assert name ∈ traj.names "Variable $name not found in trajectory"
        ts = con.times
        
        # Handle scalar value - repeat for variable dimension
        if length(con.values) == 1
            val_per_time = fill(con.values[1], traj.dims[name])
        else
            @assert length(con.values) == traj.dims[name] "Value dimension mismatch for variable $name"
            val_per_time = con.values
        end
        
        # Apply constraint at each time step
        for t ∈ ts
            indices = slice(t, traj.components[name], traj.dim)
            for (i, val) ∈ zip(indices, val_per_time)
                MOI.add_constraints(opt, vars[i], MOI.EqualTo(val))
            end
        end
    end
end

function (con::BoundsConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
)
    name = con.var_names
    bounds_val = con.bounds_values
    
    if con.is_global
        # Global variable constraint
        @assert name ∈ traj.global_names "Global variable $name not found in trajectory"
        var_dim = traj.global_dims[name]
        
        # Convert bounds to (lb, ub) tuple vectors
        if bounds_val isa Float64
            @assert bounds_val >= 0 "Scalar bound must be non-negative"
            lb = fill(-bounds_val, var_dim)
            ub = fill(bounds_val, var_dim)
        elseif bounds_val isa Vector{Float64}
            @assert length(bounds_val) == var_dim "Bound dimension mismatch"
            @assert all(bounds_val .>= 0) "Vector bound must be non-negative"
            lb = -bounds_val
            ub = bounds_val
        else  # Tuple
            lb, ub = bounds_val
            @assert length(lb) == length(ub) == var_dim "Bound dimension mismatch"
            @assert all(lb .<= ub) "Lower bounds must be <= upper bounds"
        end
        
        indices = traj.dim * traj.N .+ traj.global_components[name]
        
        for (i, (lb_i, ub_i)) ∈ zip(indices, zip(lb, ub))
            MOI.add_constraints(opt, vars[i], MOI.GreaterThan(lb_i))
            MOI.add_constraints(opt, vars[i], MOI.LessThan(ub_i))
        end
    else
        # Trajectory variable constraint
        @assert name ∈ traj.names "Variable $name not found in trajectory"
        ts = con.times
        var_dim = traj.dims[name]
        
        # Determine subcomponents to constrain
        subcomps = isnothing(con.subcomponents) ? (1:var_dim) : con.subcomponents
        
        # Convert bounds to (lb, ub) tuple vectors
        if bounds_val isa Float64
            @assert bounds_val >= 0 "Scalar bound must be non-negative"
            lb = fill(-bounds_val, var_dim)
            ub = fill(bounds_val, var_dim)
        elseif bounds_val isa Vector{Float64}
            @assert length(bounds_val) == var_dim "Bound dimension mismatch"
            @assert all(bounds_val .>= 0) "Vector bound must be non-negative"
            lb = -bounds_val
            ub = bounds_val
        else  # Tuple
            lb, ub = bounds_val
            @assert length(lb) == length(ub) == var_dim "Bound dimension mismatch"
            @assert all(lb .<= ub) "Lower bounds must be <= upper bounds"
        end
        
        # Apply bounds at each time step (only for selected subcomponents)
        for t ∈ ts
            indices = slice(t, traj.components[name][subcomps], traj.dim)
            for (i, (lb_i, ub_i)) ∈ zip(indices, zip(lb[subcomps], ub[subcomps]))
                MOI.add_constraints(opt, vars[i], MOI.GreaterThan(lb_i))
                MOI.add_constraints(opt, vars[i], MOI.LessThan(ub_i))
            end
        end
    end
end

function (con::AllEqualConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
)
    # Determine which variable to constrain (use trajectory's timestep if :Δt)
    var_name = con.var_name == :Δt ? traj.timestep : con.var_name
    @assert var_name isa Symbol "Trajectory must have a timestep variable for AllEqualConstraint"
    @assert var_name ∈ traj.names "Variable $var_name not found in trajectory"
    
    comp_idx = con.component_index
    @assert comp_idx <= traj.dims[var_name] "Component index $comp_idx exceeds variable dimension"
    
    # All timesteps 1:N-1 must equal timestep N
    indices = [index(k, traj.components[var_name][comp_idx], traj.dim) for k ∈ 1:traj.N-1]
    bar_index = index(traj.N, traj.components[var_name][comp_idx], traj.dim)
    
    x_minus_val = MOI.ScalarAffineTerm(-1.0, vars[bar_index])
    for i ∈ indices
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
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
)
    v_comps = traj.components[con.var_name]
    s_comps = traj.components[con.slack_name]

    for t ∈ con.times
        v_indices = slice(t, v_comps, traj.dim)
        s_indices = slice(t, s_comps, traj.dim)

        for (vi, si) ∈ zip(v_indices, s_indices)
            # v_{k,i} - s_{k,i} ≤ 0
            MOI.add_constraints(
                opt,
                MOI.ScalarAffineFunction([
                    MOI.ScalarAffineTerm(1.0, vars[vi]),
                    MOI.ScalarAffineTerm(-1.0, vars[si])
                ], 0.0),
                MOI.LessThan(0.0)
            )
            # -v_{k,i} - s_{k,i} ≤ 0
            MOI.add_constraints(
                opt,
                MOI.ScalarAffineFunction([
                    MOI.ScalarAffineTerm(-1.0, vars[vi]),
                    MOI.ScalarAffineTerm(-1.0, vars[si])
                ], 0.0),
                MOI.LessThan(0.0)
            )
        end
    end
end

function (con::TotalConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
)
    # Determine which variable to sum (use trajectory's timestep if :Δt)
    var_name = con.var_name == :Δt ? traj.timestep : con.var_name
    @assert var_name isa Symbol "Trajectory must have a timestep variable for TotalConstraint"
    @assert var_name ∈ traj.names "Variable $var_name not found in trajectory"
    
    comp_idx = con.component_index
    @assert comp_idx <= traj.dims[var_name] "Component index $comp_idx exceeds variable dimension"
    
    # For timestep variables, sum only first N-1 (last knot point has no duration after it)
    # For other variables, sum all N values
    time_indices = (var_name == traj.timestep) ? (1:traj.N-1) : (1:traj.N)
    indices = [index(k, traj.components[var_name][comp_idx], traj.dim) for k ∈ time_indices]
    
    MOI.add_constraints(
        opt,
        MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, vars[idx]) for idx in indices], 0.0),
        MOI.EqualTo(con.value)
    )
end

function (con::SymmetryConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
)
    @assert con.var_name ∈ traj.names "Variable $(con.var_name) not found in trajectory"
    
    even_pairs = Vector{Tuple{Int,Int}}()
    odd_pairs = Vector{Tuple{Int,Int}}()
    
    # Get component indices for the variable
    component_indices = [slice(t, traj.components[con.var_name], traj.dim)[con.component_indices] for t ∈ 1:traj.N]
    
    if con.even
        even_pairs = vcat(even_pairs, reduce(vcat, [
            collect(zip(component_indices[[idx, traj.N - idx + 1]]...)) 
            for idx in 1:traj.N ÷ 2
        ]))
    else 
        odd_pairs = vcat(odd_pairs, reduce(vcat, [
            collect(zip(component_indices[[idx, traj.N - idx + 1]]...)) 
            for idx in 1:traj.N ÷ 2
        ]))
    end 
    
    # Add timestep symmetry if requested and timestep exists
    if con.include_timestep && traj.timestep isa Symbol
        time_indices = [index(k, traj.components[traj.timestep][1], traj.dim) for k ∈ 1:traj.N]
        even_pairs = vcat(even_pairs, [
            (time_indices[idx], time_indices[traj.N + 1 - idx]) 
            for idx ∈ 1:traj.N÷2
        ]) 
    end 
    
    # Add even symmetry constraints: x[t] = x[N-t+1]
    for (i1, i2) in even_pairs
        MOI.add_constraints(
            opt,
            MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, vars[i1]), MOI.ScalarAffineTerm(-1.0, vars[i2])], 0.0),
            MOI.EqualTo(0.0)
        )
    end
    
    # Add odd symmetry constraints: x[t] = -x[N-t+1]
    for (i1, i2) in odd_pairs
        MOI.add_constraints(
            opt,
            MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, vars[i1]), MOI.ScalarAffineTerm(1.0, vars[i2])], 0.0),
            MOI.EqualTo(0.0)
        )
    end
end

function (con::TimeConsistencyConstraint)(
    opt::Ipopt.Optimizer,
    vars::Vector{MOI.VariableIndex},
    traj::NamedTrajectory
)
    # Get variable names, using trajectory's timestep if :Δt is specified
    time_name = con.time_name
    timestep_name = con.timestep_name == :Δt ? traj.timestep : con.timestep_name
    
    @assert time_name ∈ traj.names "Time variable $time_name not found in trajectory"
    @assert timestep_name isa Symbol "Trajectory must have a timestep variable"
    @assert timestep_name ∈ traj.names "Timestep variable $timestep_name not found in trajectory"
    
    # For each k = 1:N-1, add constraint: t_{k+1} - t_k - Δt_k = 0
    for k in 1:traj.N-1
        t_k = index(k, traj.components[time_name][1], traj.dim)
        t_k1 = index(k+1, traj.components[time_name][1], traj.dim)
        Δt_k = index(k, traj.components[timestep_name][1], traj.dim)
        
        # t_{k+1} - t_k - Δt_k = 0
        MOI.add_constraints(
            opt,
            MOI.ScalarAffineFunction([
                MOI.ScalarAffineTerm(1.0, vars[t_k1]),   # + t_{k+1}
                MOI.ScalarAffineTerm(-1.0, vars[t_k]),   # - t_k
                MOI.ScalarAffineTerm(-1.0, vars[Δt_k])   # - Δt_k
            ], 0.0),
            MOI.EqualTo(0.0)
        )
    end
end