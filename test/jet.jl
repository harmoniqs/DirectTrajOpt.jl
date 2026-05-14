# JET integrates tightly with the Julia compiler and is gated to v1.12 only.
# On 1.10 / 1.11, JET 0.10 may not surface the same findings, which would
# break `broken = true` with "Unexpected Pass". Don't relax this gate.
#
# Performance analysis (type instabilities / runtime dispatch) — run manually:
#     julia --project=. -e 'using DirectTrajOpt, JET; JET.@report_opt some_function(...)'

@testitem "JET correctness analysis" tags=[:jet] begin
    if VERSION >= v"1.12"
        using JET, DirectTrajOpt
        # Findings as of initial wiring (DirectTrajOpt v0.9.3, JET 0.11):
        # 18 findings dominated by:
        #   - `show_diffs(::SparseVector, ::Any)` no-matching-method when the
        #     sparse-Jacobian union-split lands on the SparseVector branch.
        #   - `randn(::Int64, ::NTuple{N,Int64})` test-utility helper that
        #     forwards a tuple as the trailing dim argument.
        # TODO: address in a follow-up PR; drop `broken = true` once clean.
        JET.test_package(DirectTrajOpt; target_modules = (DirectTrajOpt,), broken = true)
    else
        @info "Skipping JET correctness analysis on Julia $VERSION (requires >= 1.12)"
        @test true
    end
end
