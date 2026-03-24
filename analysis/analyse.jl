#!/usr/bin/env julia
# analyse.jl — Growth curve analysis for E. coli Keio knockout dataset
#
# Setup (run once):
#   julia --project=. -e '
#     using Pkg
#     Pkg.develop(path="../../KinBiont.jl")
#     Pkg.instantiate()
#   '
#
# Run:
#   julia --project=. analyse.jl
#
# Output: ../docs/data/curves_data.json

using XLSX
using Statistics
using JSON3
using Kinbiont

const DATA_DIR = joinpath(@__DIR__, "..")
const OUT_PATH = joinpath(DATA_DIR, "docs", "data", "curves_data.json")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_as_float(v) = (ismissing(v) || v === nothing) ? NaN : Float64(v)
_as_str(v)   = (ismissing(v) || v === nothing) ? ""  : string(v)

function find_elbow(ks, wcss_vals)
    # Largest second-difference (maximum curvature in the elbow)
    length(ks) < 3 && return ks[end]
    d2 = diff(diff(wcss_vals))
    return ks[argmax(d2) + 1]
end

function raw_centroids(curves::Matrix{Float64}, labels::Vector{Int}, n_k::Int)
    n_tp = size(curves, 2)
    cents = zeros(n_k, n_tp)
    for k in 1:n_k
        idx = findall(==(k), labels)
        isempty(idx) && continue
        cents[k, :] = vec(mean(curves[idx, :], dims=1))
    end
    return cents
end

# ─────────────────────────────────────────────────────────────────────────────
# 1. Metadata: curve_id → (gene, jw_id, medium)
# ─────────────────────────────────────────────────────────────────────────────

function load_metadata()
    path = joinpath(DATA_DIR, "Curves_knockouts_media.xlsx")
    @info "Reading metadata from $path"
    xf = XLSX.readxlsx(path)
    sh = xf[1]
    data = sh[:]

    meta = Dict{String, @NamedTuple{gene::String, jw_id::String, medium::String}}()
    for i in 2:size(data, 1)
        row = data[i, :]
        any(j -> ismissing(row[j]), 1:5) && continue
        curve_id  = _as_str(row[1])
        jw_id     = _as_str(row[2])
        gene_name = _as_str(row[3])
        # col 4 = gene_category (unused here)
        medium    = _as_str(row[5])
        isempty(curve_id) && continue
        meta[curve_id] = (; gene=gene_name, jw_id, medium)
    end
    @info "  $(length(meta)) curve-to-gene mappings loaded"
    return meta
end

# ─────────────────────────────────────────────────────────────────────────────
# 2. Growth curves for one medium file
# ─────────────────────────────────────────────────────────────────────────────

function load_curves(path::String, sheet_name::String, wanted::Set{String})
    @info "Reading curves from $path (sheet=$sheet_name, want=$(length(wanted)) curves)"
    xf   = XLSX.readxlsx(path)
    sh   = xf[sheet_name]
    data = sh[:]
    nrows, ncols = size(data)

    # Row 1: column headers  (Time, Curve00001, …)
    headers = [_as_str(data[1, j]) for j in 1:ncols]

    # Collect valid time rows
    times     = Float64[]
    row_index = Int[]
    for i in 2:nrows
        t = _as_float(data[i, 1])
        isfinite(t) || continue
        push!(times, t)
        push!(row_index, i)
    end

    # Extract only curves in `wanted`
    curves = Dict{String, Vector{Float64}}()
    for j in 2:ncols
        hdr = headers[j]
        hdr in wanted || continue
        vals = [_as_float(data[row_index[k], j]) for k in eachindex(row_index)]
        # Keep NaN for missing/empty cells — downstream handles them
        curves[hdr] = vals
    end

    @info "  $(length(times)) time points, $(length(curves)) curves extracted"
    return times, curves
end

# ─────────────────────────────────────────────────────────────────────────────
# 3. Aggregate replicates → mean ± std per (gene, medium)
# ─────────────────────────────────────────────────────────────────────────────

function aggregate_by_gene(meta, curves_dict::Dict{String,Vector{Float64}})
    # curves_dict: curve_id → OD vector (all same medium)
    groups = Dict{String, Vector{Vector{Float64}}}()   # gene → list of replicate vectors
    jw_map = Dict{String, String}()

    for (curve_id, info) in meta
        haskey(curves_dict, curve_id) || continue
        gene = info.gene
        if !haskey(groups, gene)
            groups[gene]  = Vector{Float64}[]
            jw_map[gene]  = info.jw_id
        end
        push!(groups[gene], curves_dict[curve_id])
    end

    result = Dict{String, @NamedTuple{mean::Vector{Float64}, std::Vector{Float64},
                                       n::Int, jw_id::String}}()
    for (gene, replicates) in groups
        mat = reduce(hcat, replicates)   # n_tp × n_replicates
        # NaN-aware mean and std: ignore missing values per timepoint
        μ = map(1:size(mat, 1)) do i
            vs = filter(!isnan, mat[i, :])
            isempty(vs) ? NaN : Statistics.mean(vs)
        end
        σ = map(1:size(mat, 1)) do i
            vs = filter(!isnan, mat[i, :])
            length(vs) > 1 ? Statistics.std(vs) : 0.0
        end
        result[gene] = (; mean=μ, std=σ, n=size(mat, 2), jw_id=jw_map[gene])
    end
    return result
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. KinBiont clustering with WCSS elbow sweep
# ─────────────────────────────────────────────────────────────────────────────

function cluster_gene_curves(
    gene_means::Matrix{Float64},   # n_genes × n_tp
    times::Vector{Float64},
    gene_labels::Vector{String};
    k_range = 2:10,
)
    gd = GrowthData(gene_means, times, gene_labels)

    # WCSS sweep
    @info "  Running WCSS sweep k=$(first(k_range))..$(last(k_range))"
    ks        = collect(k_range)
    wcss_vals = Float64[]
    for k in ks
        opts      = FitOptions(
            cluster                  = true,
            n_clusters               = k,
            cluster_prescreen_constant = true,
            cluster_tol_const        = 1.5,
        )
        proc = preprocess(gd, opts)
        push!(wcss_vals, something(proc.wcss, 0.0))
    end

    # Elbow
    opt_k = find_elbow(ks, wcss_vals)
    @info "  Optimal k = $opt_k (elbow method)"

    # Final clustering
    opts_final = FitOptions(
        cluster                  = true,
        n_clusters               = opt_k,
        cluster_prescreen_constant = true,
        cluster_tol_const        = 1.5,
    )
    proc_final = preprocess(gd, opts_final)

    return (
        ks         = ks,
        wcss       = wcss_vals,
        optimal_k  = opt_k,
        clusters   = something(proc_final.clusters, ones(Int, size(gene_means, 1))),
        # z-scored centroids (shape prototypes, scale-independent)
        centroids_z  = proc_final.centroids,
        # original-space centroids
        centroids_raw = raw_centroids(gene_means, something(proc_final.clusters, ones(Int, size(gene_means, 1))), opt_k),
    )
end

# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────

function main()
    # Load metadata
    meta = load_metadata()

    # Determine which curve IDs we need per medium
    ids_lb  = Set(k for (k, v) in meta if v.medium == "LB")
    ids_m63 = Set(k for (k, v) in meta if v.medium == "M63")

    # Subset meta per medium (so aggregate_by_gene sees correct medium's curves)
    meta_lb  = Dict(k => v for (k, v) in meta if v.medium == "LB")
    meta_m63 = Dict(k => v for (k, v) in meta if v.medium == "M63")

    # Load raw curves
    times_lb, curves_lb = load_curves(
        joinpath(DATA_DIR, "Growth_curves_LB.xlsx"), "LB", ids_lb)
    times_m63, curves_m63 = load_curves(
        joinpath(DATA_DIR, "Growth_curves_M63.xlsx"), "M63", ids_m63)

    # Use the shorter common time range if they differ
    times = length(times_lb) <= length(times_m63) ? times_lb : times_m63

    # Trim curves to common length
    n_tp = length(times)
    foreach(k -> curves_lb[k]  = curves_lb[k][1:n_tp],  keys(curves_lb))
    foreach(k -> curves_m63[k] = curves_m63[k][1:n_tp], keys(curves_m63))

    # Aggregate replicates
    @info "Aggregating replicates by gene..."
    agg_lb  = aggregate_by_gene(meta_lb,  curves_lb)
    agg_m63 = aggregate_by_gene(meta_m63, curves_m63)

    # Union of genes present in both media
    genes_lb  = Set(keys(agg_lb))
    genes_m63 = Set(keys(agg_m63))
    all_genes = sort(collect(genes_lb ∪ genes_m63))
    @info "  $(length(all_genes)) unique genes (LB: $(length(genes_lb)), M63: $(length(genes_m63)))"

    # Build gene-level matrices per medium (only genes present in that medium)
    genes_lb_sorted  = sort(collect(genes_lb))
    genes_m63_sorted = sort(collect(genes_m63))

    mat_lb  = reduce(vcat, [agg_lb[g].mean'  for g in genes_lb_sorted])   # n_genes_lb  × n_tp
    mat_m63 = reduce(vcat, [agg_m63[g].mean' for g in genes_m63_sorted])  # n_genes_m63 × n_tp

    # Clustering needs finite values — replace NaN with 0 only for this input
    mat_lb_cl  = replace(mat_lb,  NaN => 0.0)
    mat_m63_cl = replace(mat_m63, NaN => 0.0)

    # Cluster
    @info "Clustering LB gene curves..."
    cl_lb  = cluster_gene_curves(mat_lb_cl,  times, genes_lb_sorted)

    @info "Clustering M63 gene curves..."
    cl_m63 = cluster_gene_curves(mat_m63_cl, times, genes_m63_sorted)

    # Downsample time axis for JSON output (full resolution used for clustering above)
    # Keep every 4th point → 200 → ~50 points: reduces JSON ~4x with negligible visual loss
    ds       = 4
    ds_idx   = 1:ds:n_tp
    times_ds = times[ds_idx]
    # NaN → nothing so JSON3 writes null (Plotly draws a gap, not a line to zero)
    ds_vec(v) = Union{Float64,Nothing}[isnan(x) ? nothing : round(x; digits=6) for x in v[ds_idx]]

    # Build per-gene JSON records
    lb_cluster_map  = Dict(zip(genes_lb_sorted,  cl_lb.clusters))
    m63_cluster_map = Dict(zip(genes_m63_sorted, cl_m63.clusters))

    # Collect unique jw_ids (prefer LB, fallback to M63)
    jw_ids = Dict{String, String}()
    for (g, info) in agg_lb;  jw_ids[g] = info.jw_id; end
    for (g, info) in agg_m63; haskey(jw_ids, g) || (jw_ids[g] = info.jw_id); end

    gene_records = []
    for gene in all_genes
        rec = Dict{String, Any}(
            "gene"  => gene,
            "jw_id" => get(jw_ids, gene, ""),
        )
        if haskey(agg_lb, gene)
            rec["LB"] = Dict(
                "mean"         => ds_vec(agg_lb[gene].mean),
                "std"          => ds_vec(agg_lb[gene].std),
                "n_replicates" => agg_lb[gene].n,
                "cluster"      => lb_cluster_map[gene],
            )
        end
        if haskey(agg_m63, gene)
            rec["M63"] = Dict(
                "mean"         => ds_vec(agg_m63[gene].mean),
                "std"          => ds_vec(agg_m63[gene].std),
                "n_replicates" => agg_m63[gene].n,
                "cluster"      => m63_cluster_map[gene],
            )
        end
        push!(gene_records, rec)
    end

    # Assemble final JSON
    out = Dict(
        "metadata" => Dict(
            "n_genes"      => length(all_genes),
            "n_timepoints" => length(times_ds),
            "media"        => ["LB", "M63"],
            "source"       => "https://www.nature.com/articles/s41597-026-07075-9",
            "description"  => "E. coli Keio knockout collection growth curves (mean of replicates per gene)",
        ),
        "times"     => round.(times_ds; digits=4),
        "wcss_sweep" => Dict(
            "LB"  => Dict("ks" => cl_lb.ks,  "wcss" => round.(cl_lb.wcss;  digits=4)),
            "M63" => Dict("ks" => cl_m63.ks, "wcss" => round.(cl_m63.wcss; digits=4)),
        ),
        "optimal_k" => Dict(
            "LB"  => cl_lb.optimal_k,
            "M63" => cl_m63.optimal_k,
        ),
        "centroids" => Dict(
            "LB"  => [ds_vec(cl_lb.centroids_raw[k, :])  for k in 1:cl_lb.optimal_k],
            "M63" => [ds_vec(cl_m63.centroids_raw[k, :]) for k in 1:cl_m63.optimal_k],
        ),
        "centroids_z" => Dict(
            "LB"  => [ds_vec(cl_lb.centroids_z[k, :])  for k in 1:cl_lb.optimal_k],
            "M63" => [ds_vec(cl_m63.centroids_z[k, :]) for k in 1:cl_m63.optimal_k],
        ),
        "genes" => gene_records,
        # With cluster_prescreen_constant=true, KinBiont always assigns
        # non-growing wells to label n_clusters (the last index).
        "nongrowing_cluster" => Dict(
            "LB"  => cl_lb.optimal_k,
            "M63" => cl_m63.optimal_k,
        ),
        "nongrowing_genes" => Dict(
            "LB"  => sort([g for g in genes_lb_sorted  if lb_cluster_map[g]  == cl_lb.optimal_k]),
            "M63" => sort([g for g in genes_m63_sorted if m63_cluster_map[g] == cl_m63.optimal_k]),
        ),
    )

    mkpath(dirname(OUT_PATH))
    open(OUT_PATH, "w") do io
        JSON3.write(io, out)
    end
    @info "Output written to $OUT_PATH"
end

main()
