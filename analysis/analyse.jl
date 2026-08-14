#!/usr/bin/env julia
# analyse.jl — Growth curve analysis for E. coli Keio knockout dataset
#
# Setup (run once):
#   julia --project=. -e 'import Pkg; Pkg.instantiate()'
#
# Kinbiont is pinned to the registered v1.5.1 (Project.toml compat,
# https://github.com/pinheiroGroup/Kinbiont.jl) — no local dev path.
#
# Run:
#   julia --project=. analyse.jl
#
# Outputs: ../docs/data/curves_data.json,
#          ../results/keio_s1_metadata.csv and
#          ../results/keio_m63_nongrowing_genes.json

using XLSX
using CSV
using DataFrames
using Statistics
using JSON3
using Kinbiont

const DATA_DIR    = joinpath(@__DIR__, "../data")
const RESULTS_DIR = joinpath(@__DIR__, "../results")
# The tracked file the frontend and analysis/enrichment_nongrowers.py read.
# This used to resolve under DATA_DIR, i.e. ../data/docs/data/, so runs wrote to
# an untracked copy and every consumer kept reading a stale checked-in file.
const OUT_PATH    = joinpath(@__DIR__, "..", "docs", "data", "curves_data.json")
const NONGROWING_JSON_PATH = joinpath(RESULTS_DIR, "keio_m63_nongrowing_genes.json")
const S1_METADATA_PATH = joinpath(RESULTS_DIR, "keio_s1_metadata.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_as_float(v) = (ismissing(v) || v === nothing) ? NaN : Float64(v)
_as_str(v)   = (ismissing(v) || v === nothing) ? ""  : string(v)

# Load the replicate-averaged matrix that scripts/01_build_gene_means.py writes.
#
# Clustering (and therefore the non-growing pre-screen) must run on these curves,
# not on a matrix re-aggregated here from the raw workbooks. The Python builder
# extends every replicate to the full run before averaging (leading values back-
# filled, internal gaps carried forward, post-final values carried forward), so
# each strain is judged over the whole 0.25–50 h window. Re-aggregating instead
# truncates each strain at the earliest replicate dropout, which cuts late
# growers short and makes them look flat: strains such as ptsH, ybhM and yccE
# reach OD 0.6–0.8 only after 23 h and were previously mis-assigned to the
# non-growing sentinel. `genes` fixes the row order.
function load_gene_mean_matrix(
    medium::String,
    genes::Vector{String},
)::Tuple{Vector{Float64}, Matrix{Float64}}
    path = joinpath(RESULTS_DIR, "keio_$(lowercase(medium))_gene_means.csv")
    isfile(path) || error(
        "Replicate-averaged matrix not found: $path\n" *
        "Run `python scripts/01_build_gene_means.py` first."
    )
    df    = CSV.read(path, DataFrame)
    times = Float64.(df[!, 1])

    missing_genes = filter(g -> !hasproperty(df, Symbol(g)), genes)
    isempty(missing_genes) || error(
        "$medium: $(length(missing_genes)) gene(s) absent from $path: " *
        join(first(missing_genes, 10), ", ")
    )

    mat = Matrix{Float64}(undef, length(genes), length(times))
    for (i, g) in enumerate(genes)
        mat[i, :] = Float64.(df[!, Symbol(g)])
    end
    return times, mat
end

# Emit the centroid table Figure 2c reads: one row per cluster per scale, with
# the cluster size and the centroid sampled on the clustering grid.
const CENTROID_CSV_DIR = get(ENV, "GUIBIONT_PAPER_SCRIPTS_DIR",
    joinpath(@__DIR__, "..", "..", "GUIbiontPaper", "scripts"))

function write_centroid_csv(medium::String, times, curves, cl)
    if !isdir(CENTROID_CSV_DIR)
        @warn "  Centroid table not written: $CENTROID_CSV_DIR does not exist " *
              "(set GUIBIONT_PAPER_SCRIPTS_DIR to override)"
        return
    end
    zs   = Kinbiont._zscore_rows(curves)
    cols = ["centroid_t_$(round(t; digits=4))" for t in times]
    rows = NamedTuple[]
    # Row order matches the tables committed to the paper repo so a rerun diffs
    # cleanly against them. Values are rounded to 12 digits rather than written
    # at full precision: `mean` sums in a BLAS-thread-dependent order, so the
    # last bit of a Float64 is not reproducible across machines, and a
    # full-precision table shows spurious 1-ULP differences on a rerun. Twelve
    # digits is stable across configurations and still six orders of magnitude
    # finer than the 0.000001 this previously rounded to.
    for k in 1:cl.optimal_k
        idx = findall(==(k), cl.clusters)
        isempty(idx) && continue
        for scale in ("raw", "normalized")
            src = scale == "raw" ? curves : zs
            c = vec(mean(src[idx, :], dims=1))
            push!(rows, (; cluster=k, n_series=length(idx), type=scale,
                         NamedTuple{Tuple(Symbol.(cols))}(Tuple(round.(c; digits=12)))...))
        end
    end
    path = joinpath(CENTROID_CSV_DIR, "cluster_centroids_raw_and_normalized_$(medium).csv")
    CSV.write(path, DataFrame(rows))
    @info "  Centroid table written to $path"
end

# Elbow diagnostic: the k maximizing the second finite difference of the WCSS
# curve (Guibiont.tex Methods, "Elbow support for choosing k"). k=1 never
# reserves a non-growing sentinel, so its WCSS only excludes what every k>=2
# candidate also excludes when the pre-screen finds nothing to exclude at any
# k (e.g. LB) — there k=1 stays a valid neighbor. When some k>=2 DOES carve
# out a sentinel (e.g. M63), WCSS(1) includes curves WCSS(k) excludes, so k=1
# cannot serve as a neighbor: candidates start at k=3 instead of k=2. Mirrors
# GUIbiont's static/js/clustering.js:_detectElbow.
function find_elbow(ks::Vector{Int}, wcss_vals::Vector{Float64}, n_nongrowing::Vector{Int})
    n = length(wcss_vals)
    n < 3 && return ks[1]
    any_excluded = any(>(0), n_nongrowing)
    start = any_excluded ? 3 : 2
    start > n - 1 && (start = 2)
    max_d2 = -Inf
    elbow_idx = start
    for i in start:(n - 1)
        d2 = wcss_vals[i - 1] - 2 * wcss_vals[i] + wcss_vals[i + 1]
        if d2 > max_d2
            max_d2 = d2
            elbow_idx = i
        end
    end
    return ks[elbow_idx]
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
# 3. Aggregate replicates → mean ± SEM per (gene, medium)
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

    result = Dict{String, @NamedTuple{mean::Vector{Float64}, sem::Vector{Float64},
                                       n::Int, jw_id::String,
                                       replicates::Vector{Vector{Float64}}}}()
    for (gene, replicates) in groups
        mat = reduce(hcat, replicates)   # n_tp × n_replicates
        n_rep = size(mat, 2)

        # Per-replicate valid range: last timepoint where each replicate has finite data.
        # The gene mean is only valid up to the minimum last-valid index across all
        # replicates; beyond that, one or more replicates have dropped out and the mean
        # would be biased by whichever replicate happens to run longest.
        last_valid = n_rep > 1 ? minimum(
            let li = findlast(isfinite, mat[:, j])
                li === nothing ? 0 : li
            end
            for j in 1:n_rep
        ) : size(mat, 1)

        μ = map(1:size(mat, 1)) do i
            i > last_valid && return NaN
            vs = filter(!isnan, mat[i, :])
            isempty(vs) ? NaN : Statistics.mean(vs)
        end
        # SEM (std / sqrt(n)) shows uncertainty in the mean (more meaningful at n=3).
        # NaN beyond last_valid so interpolation respects the gene's true range.
        sem = map(1:size(mat, 1)) do i
            i > last_valid && return NaN
            vs = filter(!isnan, mat[i, :])
            isempty(vs) ? NaN :
            length(vs) > 1 ? Statistics.std(vs) / sqrt(length(vs)) : 0.0
        end
        result[gene] = (; mean=μ, sem=sem, n=n_rep, jw_id=jw_map[gene],
                          replicates=[vec(mat[:, j]) for j in 1:n_rep])
    end
    return result
end

# ─────────────────────────────────────────────────────────────────────────────
# 4. Kinbiont clustering with WCSS elbow sweep
# ─────────────────────────────────────────────────────────────────────────────

# Identify the non-growing gene-level profiles with Kinbiont's own detector. Per SM.tex
# Methods ("Trajectory preparation for clustering"), the same complete
# 0.25–50 h replicate-averaged trajectories are used for both this pre-screen
# and shape clustering below — there is no separate truncated grid.
function detect_non_growing(
    gene_means_full::Matrix{Float64},   # n_genes × n_tp, full length, finite
    times_full::Vector{Float64},
    gene_labels::Vector{String};
    tol_const::Float64 = 0.5,      # τ (Kinbiont default; SM.tex Methods)
    q_low::Float64  = 0.05,        # lower pre-screen quantile (Kinbiont default)
    q_high::Float64 = 0.95,        # upper pre-screen quantile (Kinbiont default)
)::Vector{Bool}
    idx = detect_non_growing_indices(
        gene_means_full, times_full;
        prescreen_constant = true,
        prescreen_tol      = tol_const,
        prescreen_q_low    = q_low,
        prescreen_q_high   = q_high,
    )
    mask = falses(length(gene_labels))
    mask[idx] .= true
    return collect(mask)
end

# Cluster gene curves on the full 0.25–50 h grid, delegating entirely to
# Kinbiont's own prescreen-aware clustering — the same `preprocess`/
# `FitOptions` call GUIbiont's `/api/cluster` route makes — so sentinel
# reservation, WCSS exclusion and z-scoring can never drift from what
# GUIbiont actually computes. No separate truncated grid and no post-hoc
# reassignment of late-growing trajectories (SM.tex Methods).
function cluster_gene_curves(
    gene_means::Matrix{Float64},   # n_genes × n_tp, full 0.25–50 h grid
    times::Vector{Float64},
    gene_labels::Vector{String};
    k::Int,                        # final k (includes the sentinel, if any)
    tol_const::Float64 = 0.5,
    q_low::Float64  = 0.05,
    q_high::Float64 = 0.95,
    k_max_sweep::Int = 10,
)
    n_genes = size(gene_means, 1)
    gd = GrowthData(gene_means, times, gene_labels)

    non_growing_idx = detect_non_growing_indices(
        gene_means, times;
        prescreen_constant = true,
        prescreen_tol      = tol_const,
        prescreen_q_low    = q_low,
        prescreen_q_high   = q_high,
    )
    # Only actually request the sentinel from Kinbiont when the criterion
    # found something to reserve one for — matching GUIbiont's `do_prescreen`
    # (routes/ml.jl) — otherwise the "last" cluster label is just an ordinary
    # k-means cluster, not a sentinel, and must not be reported as one.
    prescreen = !isempty(non_growing_idx)
    @info "  $(length(non_growing_idx)) non-growing gene(s) detected (τ=$tol_const, q=$q_low/$q_high)"

    # cluster_trend_test defaults to `true` in Kinbiont's FitOptions — it must
    # be explicitly disabled here, matching GUIbiont's routes/ml.jl (which
    # always passes it explicitly from the trend-test checkbox state), or the
    # slope trend test silently reserves extra sentinel slots on top of the
    # prescreen: it flagged secG, metH and ybaO on M63 even though none of
    # them are remotely flat (OD range 0.8–1.0), because the paper's Methods
    # only use the constant-curve pre-screen, not the trend test.
    common = (cluster_method = :kmeans, cluster_tol_const = tol_const,
              cluster_q_low = q_low, cluster_q_high = q_high,
              cluster_trend_test = false,
              kmeans_seed = 42, kmedoids_seed = 42,
              kmeans_n_init = 3, kmedoids_n_init = 3,
              kmeans_max_iters = 300, kmeans_tol = 1e-6)

    # Informational WCSS sweep k=1..k_max_sweep, mirroring GUIbiont's
    # /api/cluster-sweep: k=1 never reserves a sentinel (single-cluster
    # baseline), k>=2 does when the pre-screen has anything to reserve.
    @info "  Running WCSS sweep k=1..$k_max_sweep (prescreen=$prescreen)"
    ks = collect(1:k_max_sweep)
    wcss_vals = Float64[]
    n_nongrowing_sweep = Int[]
    for kk in ks
        prescreen_for_k = prescreen && kk > 1
        proc = preprocess(gd, FitOptions(;
            cluster = true, n_clusters = kk,
            cluster_prescreen_constant = prescreen_for_k,
            common...))
        push!(wcss_vals, something(proc.wcss, 0.0))
        push!(n_nongrowing_sweep, prescreen_for_k ? length(non_growing_idx) : 0)
    end
    auto_k = find_elbow(ks, wcss_vals, n_nongrowing_sweep)
    @info "  Optimal k = $k (auto-elbow suggested $auto_k)"

    # Final clustering at the chosen k.
    proc_final = preprocess(gd, FitOptions(;
        cluster = true, n_clusters = k,
        cluster_prescreen_constant = prescreen && k > 1,
        common...))
    clusters = something(proc_final.clusters, ones(Int, n_genes))

    return (
        ks         = ks,
        wcss       = wcss_vals,
        optimal_k  = k,
        prescreen  = prescreen,
        clusters   = clusters,
        # z-scored centroids (shape prototypes, scale-independent). z-scoring
        # uses Kinbiont's own row normalization, including its constant-curve
        # guard.
        centroids_z   = raw_centroids(Kinbiont._zscore_rows(gene_means), clusters, k),
        # original-space centroids
        centroids_raw = raw_centroids(gene_means, clusters, k),
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

    # Aggregate replicates (each medium uses its own full time vector)
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

    # ── Clustering ────────────────────────────────────────────────────────────
    # Both the non-growing pre-screen and shape clustering run on the same
    # complete 0.25–50 h replicate-averaged trajectories from
    # scripts/01_build_gene_means.py — no separate truncated grid (SM.tex
    # Methods, "Trajectory preparation for clustering").
    @info "Loading full-length gene-mean matrices..."
    times_lb_full,  mat_lb_full  = load_gene_mean_matrix("LB",  genes_lb_sorted)
    times_m63_full, mat_m63_full = load_gene_mean_matrix("M63", genes_m63_sorted)

    ng_lb  = detect_non_growing(mat_lb_full,  times_lb_full,  genes_lb_sorted)
    ng_m63 = detect_non_growing(mat_m63_full, times_m63_full, genes_m63_sorted)
    @info "  LB: $(count(ng_lb)) non-growing; M63: $(count(ng_m63)) non-growing (τ=0.5, q=0.05/0.95)"

    # Paper configuration: LB has no non-growing subpopulation, so no sentinel is
    # reserved and two dynamic clusters are used; M63 reserves the last cluster
    # index for the pre-screened strains, giving two dynamic clusters plus the
    # sentinel (k=3).
    @info "Clustering LB gene curves (k=2)..."
    cl_lb  = cluster_gene_curves(mat_lb_full,  times_lb_full,  genes_lb_sorted; k = 2)

    @info "Clustering M63 gene curves (k=3)..."
    cl_m63 = cluster_gene_curves(mat_m63_full, times_m63_full, genes_m63_sorted; k = 3)

    times_lb_cl, times_m63_cl = times_lb_full, times_m63_full
    mat_lb_cl,   mat_m63_cl   = mat_lb_full,   mat_m63_full

    # ── Visualisation ─────────────────────────────────────────────────────────
    # Per-gene curves are stored on the ORIGINAL time axis (downsampled 4x).
    # Each gene has NaN where its replicates had no data — the frontend draws
    # the curve only up to its individual valid range.
    lb_cluster_map  = Dict(zip(genes_lb_sorted,  cl_lb.clusters))
    m63_cluster_map = Dict(zip(genes_m63_sorted, cl_m63.clusters))

    # NaN → nothing so JSON3 writes null (Plotly draws a gap)
    to_json_vec(v) = Union{Float64,Nothing}[isnan(x) ? nothing : round(x; digits=6) for x in v]

    # Downsample original time axes 4x for JSON output
    ds = 4
    lb_ds_idx  = 1:ds:length(times_lb)
    m63_ds_idx = 1:ds:length(times_m63)
    times_lb_ds  = times_lb[lb_ds_idx]
    times_m63_ds = times_m63[m63_ds_idx]

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
                "mean"         => to_json_vec(agg_lb[gene].mean[lb_ds_idx]),
                "sem"          => to_json_vec(agg_lb[gene].sem[lb_ds_idx]),
                "n_replicates" => agg_lb[gene].n,
                "cluster"      => lb_cluster_map[gene],
                "replicates"   => [to_json_vec(rep[lb_ds_idx]) for rep in agg_lb[gene].replicates],
            )
        end
        if haskey(agg_m63, gene)
            rec["M63"] = Dict(
                "mean"         => to_json_vec(agg_m63[gene].mean[m63_ds_idx]),
                "sem"          => to_json_vec(agg_m63[gene].sem[m63_ds_idx]),
                "n_replicates" => agg_m63[gene].n,
                "cluster"      => m63_cluster_map[gene],
                "replicates"   => [to_json_vec(rep[m63_ds_idx]) for rep in agg_m63[gene].replicates],
            )
        end
        push!(gene_records, rec)
    end

    nongrowing_genes = Dict(
        "LB"  => cl_lb.prescreen  ? sort([g for g in genes_lb_sorted  if lb_cluster_map[g]  == cl_lb.optimal_k])  : String[],
        "M63" => cl_m63.prescreen ? sort([g for g in genes_m63_sorted if m63_cluster_map[g] == cl_m63.optimal_k]) : String[],
    )

    # Lightweight provenance table used by Supplementary Data S1. These are
    # metadata of the gene-level aggregation, so reproducing S1 must not require
    # rerunning the unrelated four-model parametric fit in batch_fit.jl.
    metadata_rows = NamedTuple[]
    for gene in all_genes
        for (medium, agg) in (("LB", agg_lb), ("M63", agg_m63))
            haskey(agg, gene) || continue
            push!(metadata_rows, (
                gene=gene,
                # Match the atlas JSON and the historical batch-fit CSV: a
                # gene symbol has one retained JW identifier, preferring LB
                # when multiple source deletion strains map to that symbol.
                jw_id=get(jw_ids, gene, ""),
                medium=medium,
                n_replicates=agg[gene].n,
            ))
        end
    end
    CSV.write(S1_METADATA_PATH, DataFrame(metadata_rows))
    @info "S1 metadata written to $S1_METADATA_PATH"

    # Assemble final JSON — separate time axes per medium since they may differ
    out = Dict(
        "metadata" => Dict(
            "n_genes"      => length(all_genes),
            "media"        => ["LB", "M63"],
            "source"       => "https://www.nature.com/articles/s41597-026-07075-9",
            "description"  => "E. coli Keio knockout collection growth curves (mean of replicates per gene)",
        ),
        "times_LB"  => round.(times_lb_ds;  digits=4),
        "times_M63" => round.(times_m63_ds; digits=4),
        # Keep legacy "times" key as the shorter of the two for backward compat
        "times"     => round.(length(times_lb_ds) <= length(times_m63_ds) ? times_lb_ds : times_m63_ds; digits=4),
        "wcss_sweep" => Dict(
            "LB"  => Dict("ks" => cl_lb.ks,  "wcss" => round.(cl_lb.wcss;  digits=4)),
            "M63" => Dict("ks" => cl_m63.ks, "wcss" => round.(cl_m63.wcss; digits=4)),
        ),
        "optimal_k" => Dict(
            "LB"  => cl_lb.optimal_k,
            "M63" => cl_m63.optimal_k,
        ),
        # Centroids are on their own clustering-grid time axes
        "centroid_times" => Dict(
            "LB"  => round.(times_lb_cl;  digits=4),
            "M63" => round.(times_m63_cl; digits=4),
        ),
        "centroids" => Dict(
            "LB"  => [to_json_vec(cl_lb.centroids_raw[k, :])  for k in 1:cl_lb.optimal_k],
            "M63" => [to_json_vec(cl_m63.centroids_raw[k, :]) for k in 1:cl_m63.optimal_k],
        ),
        "centroids_z" => Dict(
            "LB"  => [to_json_vec(cl_lb.centroids_z[k, :])  for k in 1:size(cl_lb.centroids_z, 1)],
            "M63" => [to_json_vec(cl_m63.centroids_z[k, :]) for k in 1:size(cl_m63.centroids_z, 1)],
        ),
        "genes" => gene_records,
        # With cluster_prescreen_constant=true, Kinbiont assigns non-growing wells
        # to label n_clusters (the last index). When the pre-screen is disabled
        # (LB), no sentinel is reserved: report cluster 0 (matches no cluster) and
        # an empty gene list so the figure draws no non-growing cluster.
        "nongrowing_cluster" => Dict(
            "LB"  => cl_lb.prescreen  ? cl_lb.optimal_k  : 0,
            "M63" => cl_m63.prescreen ? cl_m63.optimal_k : 0,
        ),
        "nongrowing_genes" => nongrowing_genes,
    )

    # Centroid tables for the paper's Figure 2c generator
    # (GUIbiontPaper/scripts/plot_keio_cluster_centroids_raw_svg_v2.py). Written
    # from the same run as the JSON so the panel can never drift from the
    # clustering it is supposed to depict.
    write_centroid_csv("lb",  times_lb_cl,  mat_lb_cl,  cl_lb)
    write_centroid_csv("m63", times_m63_cl, mat_m63_cl, cl_m63)

    mkpath(dirname(OUT_PATH))
    open(OUT_PATH, "w") do io
        JSON3.write(io, out)
    end
    @info "Output written to $OUT_PATH"

    # Small adapter consumed directly by kegg_enrichment_s2.py. Keeping this
    # beside the other analysis outputs avoids a manual conversion from the
    # nested frontend JSON and guarantees that enrichment uses this exact
    # pre-screen classification.
    mkpath(dirname(NONGROWING_JSON_PATH))
    open(NONGROWING_JSON_PATH, "w") do io
        JSON3.write(io, Dict("genes" => nongrowing_genes["M63"]))
        write(io, '\n')
    end
    @info "Non-growing gene JSON written to $NONGROWING_JSON_PATH"
end

main()
