using Documenter, DocumenterVitepress, Pkg
using DiffusionModels

#! format: off

pages = [
    "DiffusionModels.jl" => "index.md",
    "Getting Started" => [
        "Introduction" => "introduction/index.md",
        # "Overview" => "introduction/overview.md",
        # "Resources" => "introduction/resources.md",
        # "Citation" => "introduction/citation.md"
    ],
    # "Tutorials" => [
    #     "Overview" => "tutorials/index.md",
    #     "Beginner" => [
    #         "tutorials/beginner/1_Basics.md",
    #     ],
    #     "Intermediate" => [
    #         "tutorials/intermediate/1.md",
    #     ],
    #     "Advanced" => [
    #         "tutorials/advanced/1.md"
    #     ]
    # ],
    # "Manual" => [
    #     "manual/interface.md",
    # ],
    "API Reference" => [
        "DiffusionModels" => [
            "api/noise_schedules.md",
        ],
    ]
]

#! format: on

# deploy_config = Documenter.auto_detect_deploy_system()
# deploy_decision = Documenter.deploy_folder(deploy_config; repo="github.com/samb-t/DiffusionModels.jl",
#     devbranch="main", devurl="dev", push_preview=true)

makedocs(;
    sitename="DiffusionModels.jl Docs",
    authors="Sam Bond-Taylor et al.",
    clean=false, # true
    doctest=false,  # We test it in the CI, no need to run it here
    modules=[
        DiffusionModels
    ],
    linkcheck=true,
    repo="https://github.com/samb-t/DiffusionModels.jl/blob/{commit}{path}#{line}",
    format=DocumenterVitepress.MarkdownVitepress(;
        repo="github.com/samb-t/DiffusionModels.jl",
        devbranch="main",
        devurl="dev",
        deploy_url="TODO",
        md_output_path=".",
        build_vitepress=false,
        # deploy_decision,
    ),
    draft=false,
    pages,
)

deploydocs(;
    repo="github.com/samb-t/DiffusionModels.jl.git",
    push_preview=true, target="build", devbranch="main"
)
