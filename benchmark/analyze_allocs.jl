using HarmoniqsBenchmarks
using Printf

const DEFAULT_RESULTS_DIR = joinpath(@__DIR__, "results", "allocs")
results_dir() = isempty(ARGS) ? DEFAULT_RESULTS_DIR : ARGS[1]

# Noise filters — frames / types from Profile.Allocs itself or the Julia
# toplevel/runtime that do not tell us anything about user-code hotpaths.
const NOISE_FRAME_PATTERNS = [
    "Profile.Allocs",
    "gc-alloc-profiler",
    "gc-stock.c",
    "gc.c:",
    "jl_apply",
    "jl_toplevel_",
    "ijl_toplevel_",
    "jl_interpret_toplevel_thunk",
    "jl_repl_entrypoint",
    "interpreter.c",
    "_include(",
    "include_string(",
    "loading.jl",
    "client.jl",
    "_start() at sys.so",
    "ip:0x",
    "_start at ",
    " at Base.jl:",
    "true_main at jlapi.c",
    "__libc_start_main",
    "loader_exe.c",
    "jl_system_image_data",
    "macro expansion at Allocs.jl",
    "boot.jl:",
    "jl_f__call_latest",
]

const WRAPPER_FRAME_PATTERNS = [
    "alloc_profile.jl",
    "benchmark_memory!",
    "HarmoniqsBenchmarks",
]

const NOISE_TYPE_PATTERNS = [
    "Profile.Allocs",
]

_is_noise_frame(f) = any(p -> occursin(p, f), NOISE_FRAME_PATTERNS)
_is_noise_type(t)  = any(p -> occursin(p, t), NOISE_TYPE_PATTERNS)

function _first_user_frame(stack)
    for f in stack
        _is_noise_frame(f) && continue
        any(p -> occursin(p, f), WRAPPER_FRAME_PATTERNS) && continue
        return f
    end
    return isempty(stack) ? "<empty>" : stack[end]
end

_is_wrapper_frame(f) = any(p -> occursin(p, f), WRAPPER_FRAME_PATTERNS)

function top_frames(profile; k = 25, scale_to_total = true, drop_wrappers = true)
    by_frame = Dict{String, Tuple{Int, Int}}()
    for s in profile.samples
        _is_noise_type(s.type_name) && continue
        for frame in s.stacktrace
            _is_noise_frame(frame) && continue
            drop_wrappers && _is_wrapper_frame(frame) && continue
            cnt, bytes = get(by_frame, frame, (0, 0))
            by_frame[frame] = (cnt + 1, bytes + s.size_bytes)
        end
    end
    ranked = sort(collect(by_frame); by = x -> -x[2][2])[1:min(k, length(by_frame))]
    scale = scale_to_total ? (1 / profile.sample_rate) : 1.0
    println("\nTop $(length(ranked)) user frames by allocated bytes (scaled ×$(Int(scale))):")
    println(rpad("  bytes", 14), rpad("samples", 10), "frame")
    for (frame, (cnt, bytes)) in ranked
        @printf "  %-12s %-8d %s\n" _fmt_bytes(bytes * scale) cnt _truncate(frame, 140)
    end
end

function top_leaf_callsites(profile; k = 25, scale_to_total = true)
    by_leaf = Dict{String, Tuple{Int, Int}}()
    for s in profile.samples
        _is_noise_type(s.type_name) && continue
        leaf = _first_user_frame(s.stacktrace)
        cnt, bytes = get(by_leaf, leaf, (0, 0))
        by_leaf[leaf] = (cnt + 1, bytes + s.size_bytes)
    end
    ranked = sort(collect(by_leaf); by = x -> -x[2][2])[1:min(k, length(by_leaf))]
    scale = scale_to_total ? (1 / profile.sample_rate) : 1.0
    println("\nTop $(length(ranked)) leaf call sites by allocated bytes (scaled ×$(Int(scale))):")
    println(rpad("  bytes", 14), rpad("samples", 10), "leaf")
    for (leaf, (cnt, bytes)) in ranked
        @printf "  %-12s %-8d %s\n" _fmt_bytes(bytes * scale) cnt _truncate(leaf, 140)
    end
end

function top_types(profile; k = 15, scale_to_total = true)
    by_type = Dict{String, Tuple{Int, Int}}()
    for s in profile.samples
        _is_noise_type(s.type_name) && continue
        cnt, bytes = get(by_type, s.type_name, (0, 0))
        by_type[s.type_name] = (cnt + 1, bytes + s.size_bytes)
    end
    ranked = sort(collect(by_type); by = x -> -x[2][2])[1:min(k, length(by_type))]
    scale = scale_to_total ? (1 / profile.sample_rate) : 1.0
    println("\nTop $(length(ranked)) allocated types (scaled ×$(Int(scale))):")
    println(rpad("  bytes", 14), rpad("samples", 10), "type")
    for (t, (cnt, bytes)) in ranked
        @printf "  %-12s %-8d %s\n" _fmt_bytes(bytes * scale) cnt _truncate(t, 120)
    end
end

_fmt_bytes(b) = b >= 1 << 30 ? @sprintf("%.2f GB", b / (1 << 30)) :
                b >= 1 << 20 ? @sprintf("%.2f MB", b / (1 << 20)) :
                b >= 1 << 10 ? @sprintf("%.2f KB", b / (1 << 10)) :
                @sprintf("%d B", Int(round(b)))

_truncate(s, n) = length(s) <= n ? s : string(first(s, n - 1), "…")

function main()
    dir = results_dir()
    files = sort(filter(f -> endswith(f, "_allocs.jld2"), readdir(dir; join = true)))
    isempty(files) && (println("no *_allocs.jld2 files under $dir"); return)
    for path in files
        profile = load_alloc_profile(path)
        println("=" ^ 100)
        println(basename(path))
        @printf "  solver=%s  N=%d  sample_rate=%g  samples=%d  total=%s\n" profile.solver profile.N profile.sample_rate profile.total_count _fmt_bytes(profile.total_bytes)
        top_types(profile; k = 10)
        top_leaf_callsites(profile; k = 20)
        top_frames(profile; k = 20)
    end
end

main()
