module Callbacks


using ..DirectTrajOpt
using NamedTrajectories
using Ipopt


"""
    # Callbacks evaluated by Ipopt should have the following signature:
    # Note that Cint === Int32 && Cdouble === Float64

    function my_intermediate_callback(
        alg_mod::Cint,
        iter_count::Cint,
        obj_value::Float64,
        inf_pr::Float64,
        inf_du::Float64,
        mu::Float64,
        d_norm::Float64,
        regularization_size::Float64,
        alpha_du::Float64,
        alpha_pr::Float64,
        ls_trials::Cint,
    )
        # ... user code ...
        return true # or `return false` to terminate the solve.
    end
"""


"""
# Example usage:
#
# # The following solve should proceed as usual, printing the current fidelity (as computed by unitary_rollout_fidelity) once every 10 iterations, and stopping once it exceeds 0.999
# > initial = unitary_rollout_fidelity(prob.trajectory, sys)
# > cb = callback_factory(_callback_update_trajectory_factory(prob), _callback_rollout_fidelity_factory(prob, sys, unitary_rollout_fidelity; fid_thresh=0.999, freq=10))
# > solve!(prob; max_iter=100, callback=cb)
# > final = unitary_rollout_fidelity(prob.trajectory, sys)
# > @assert final > initial
# 
# # Terminating the solve manually (via Ctrl+C) will result in the final fidelity matching the initial fidelity (loss of solver progress) if _callback_update_trajectory is omitted
# > do_traj_update = false
# > initial = unitary_rollout_fidelity(prob.trajectory, sys)
# > cb = callback_factory((do_traj_update ? [_callback_update_trajectory_factory(prob)] : [])...)
# > solve!(prob; max_iter=100, callback=cb)
# > final = unitary_rollout_fidelity(prob.trajectory, sys)
# > @assert (final == initial) == (!do_traj_update)
"""


IpoptOptimizerState = NamedTuple{(:alg_mod, :iter_count, :obj_value, :inf_pr, :inf_du, :mu, :d_norm, :regularization_size, :alpha_du, :alpha_pr, :ls_trials), Tuple{Int32, Int32, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64, Int32}}

function callback_factory(callbacks...; kwargs...)
    function _callback_factory(optimizer::Ipopt.Optimizer)
        function _callback(optimizer_state...)
            # for callback in callbacks
            #     res = callback(optimizer, optimizer_state; kwargs)
            #     if !res
            #         return false
            #     end
            # end
            return all([callback(optimizer, IpoptOptimizerState(optimizer_state); kwargs) for callback in callbacks])
        end
    end
end

function _callback_say_hello_factory(msg::String)
    function _callback_say_hello(optimizer::Ipopt.Optimizer, optimizer_state::IpoptOptimizerState; kwargs...)
        println(msg)
        return true
    end
end

function _callback_stop_iteration_factory(stop_iteration::Int)
    function _callback_stop_iteration(optimizer::Ipopt.Optimizer, optimizer_state::IpoptOptimizerState; kwargs...)
        if optimizer_state.iter_count >= stop_iteration
            return false
        end
        return true
    end
end

function _callback_update_trajectory_factory(problem::DirectTrajOptProblem)
    function _callback_update_trajectory(optimizer::Ipopt.Optimizer, optimizer_state::IpoptOptimizerState; kwargs...)
        IpoptSolverExt.update_trajectory!(problem, optimizer, optimizer.list_of_variable_indices)
        return true
    end
end


"""
# Consider just storing the data field of each trajectory; we should not expect trajectory structure to change during a solve
"""
function _callback_update_trajectory_history_factory(problem::DirectTrajOptProblem, trajectories::Vector{<:NamedTrajectory})
    function _callback_update_trajectory_history(optimizer::Ipopt.Optimizer, optimizer_state::IpoptOptimizerState; kwargs...)
        push!(trajectories, deepcopy(problem.trajectory))
        return true
    end
end

"""
# WARNING: This callback expects that _callback_update_trajectory was evaluated beforehand
#          However, a custom callback can just as well do both in one go, especially if the overhead from doing a trajectory update once per iteration is undesirable
"""
function _callback_rollout_fidelity_factory(problem::DirectTrajOptProblem, system::Any, fid_fn::Function; fid_thresh=nothing, freq=1)
    function _callback_rollout_fidelity(optimizer::Ipopt.Optimizer, optimizer_state::IpoptOptimizerState; kwargs...)
        if optimizer_state.iter_count % freq != 0
            return true
        end

        fid = fid_fn(problem.trajectory, system)

        # Probably comment this out and/or customize display of fidelities
        println()
        println("Fidelity: ", fid)

        return fid_thresh isa Nothing || fid < fid_thresh
    end
end

function _callback_best_rollout_fidelity_factory(problem::DirectTrajOptProblem, system::Any, fid_fn::Function, trajectories::Dict{Int32, Any}; fid_thresh=nothing, max_trajectories=1, freq=1)
    best_fid_idxs = Int32[]
    
    function _callback_best_rollout_fidelity(optimizer::Ipopt.Optimizer, optimizer_state::IpoptOptimizerState; kwargs...)
        if optimizer_state.iter_count % freq != 0
            return true
        end

        fid = fid_fn(problem.trajectory, system)

        iter = optimizer_state.iter_count
        pushed_traj = false
        for i in 1:Int(min(length(best_fid_idxs), max_trajectories))
            if trajectories[best_fid_idxs[i]][1] < fid
                if length(best_fid_idxs) < max_trajectories
                    push!(best_fid_idxs, iter)
                    (best_fid_idxs[i], best_fid_idxs[i + 1:end]) = (best_fid_idxs[end], best_fid_idxs[i:end - 1])
                else
                    pop!(trajectories, best_fid_idxs[max_trajectories])
                    (best_fid_idxs[i], best_fid_idxs[i + 1:end]) = (iter, best_fid_idxs[i:end - 1])
                end

                push!(trajectories, Pair(iter, (fid, deepcopy(problem.trajectory))))

                pushed_traj = true
                break
            end
        end
        if !pushed_traj && length(best_fid_idxs) < max_trajectories
            push!(best_fid_idxs, iter)
            push!(trajectories, Pair(iter, (fid, deepcopy(problem.trajectory))))
        end

        # Probably comment this out and/or customize display of fidelities
        println()
        println("Fidelity: ", fid)
        # println("Best fidelity indices: ", best_fid_idxs)
        println("Best fidelities: ")
        for (k, (v, _)) in trajectories
            println(k, ": ", v)
        end

        return fid_thresh isa Nothing || fid < fid_thresh
    end
end


end
