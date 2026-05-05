using DirectTrajOpt
using PiccoloDocsTemplate

pages = [
    "Home" => "index.md",
    "Getting Started" => [
        "Quickstart" => "generated/quickstart.md",
        "Complete Example" => "generated/example.md",
    ],
    "Core Concepts" => [
        "Problem Formulation" => "generated/concepts/problem_formulation.md",
        "Trajectories" => "generated/concepts/trajectories.md",
        "Integrators" => "generated/concepts/integrators.md",
        "Objectives" => "generated/concepts/objectives.md",
        "Constraints" => "generated/concepts/constraints.md",
    ],
    "Tutorials" => [
        "Linear System" => "generated/tutorials/linear_system.md",
        "Bilinear Control" => "generated/tutorials/bilinear_control.md",
        "Minimum Time" => "generated/tutorials/minimum_time.md",
    ],
    "Library" => "lib.md",
]

generate_docs(
    @__DIR__,
    "DirectTrajOpt",
    DirectTrajOpt,
    pages;
    format_kwargs = (
        canonical = "https://docs.harmoniqs.co/DirectTrajOpt.jl",
        size_threshold = 400 * 2^10,       # 400 KiB (matches Piccolo.jl)
        size_threshold_warn = 200 * 2^10,   # 200 KiB
        example_size_threshold = 16 * 2^10, # 16 KiB
    ),
)
