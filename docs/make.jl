using DirectTrajOpt

pages = [
    "Home" => "index.md",
    "Lib" => "lib.md",
    "Explanation" => "generated/explanation.md",
]

# Check if utils.jl exists and warn if not found
utils_path = joinpath(@__DIR__, "utils.jl")
if !isfile(utils_path)
    error("docs/utils.jl is required but not found. Please run get_docs_utils.sh")
end

include("utils.jl")

generate_docs(
    @__DIR__,
    "DirectTrajOpt",
    DirectTrajOpt,
    pages;
    format_kwargs = (canonical = "https://docs.harmoniqs.co/DirectTrajOpt.jl",),
)