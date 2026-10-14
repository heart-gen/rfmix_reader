#!/usr/bin/env Rscript
# Run LoGicAl (https://github.com/student-jw/LoGicAl) over a bundle written by
# rfmix_reader's `to_logical` (`rfmix-reader export-logical`).
#
#   Rscript logical_run.R --bundle <dir> --method pg|ag|pl|al|phase_pg|phase_ag
#                         [--files 0,1,...] [--out <tsv>] [--min-n 20]
#                         [--maxiter 1000] [--tol 1e-7] [--threads 1]
#
# Needs the R packages `arrow` and `jsonlite`; `LoGicAl` (plus SQUAREM) for
# every method except `phase_ag`, which is computed here in closed form.
#
# Bundle layout, missing-value encoding and the meaning of every column are
# in the bundle's manifest.json and in the rfmix_reader documentation.
# Frequencies are for the VCF ALT allele.
#
# Why `phase_ag` is not delegated to LoGicAl 0.1.0: run_logical(method =
# "phase_ag") refers to `npop`, which that function defines only when `par0`
# is NULL, and when `par0` is NULL the label-based methods take K from the
# labels present at the site (length(unique(c(z1, z2)))), which is wrong at
# any site where one ancestry has no haplotype.  Both are reported upstream;
# this script always passes `par0 = rep(1/K, K)` and computes `phase_ag`
# itself.  The closed form is the fraction of ALT alleles among phased
# haplotypes carrying each label, which is what `fixptfn_phase_pg` converges
# to when every posterior is 0/1.  It is evaluated per haplotype, so an
# individual with one unlabeled haplotype still contributes the other; the EM
# methods take whole individuals and drop any with a missing input.
#
# LoGicAl 0.1.0's NAMESPACE also omits the `importFrom(SQUAREM, squarem)` its
# roxygen header declares, so `run_logical` only finds `squarem` when SQUAREM is
# attached; this script attaches it.
#
# Output columns: n_used (individuals that contributed), n_haplotypes
# (haplotypes that contributed: 2 x n_used for the EM methods), one
# <ancestry>_freq per ancestry, and error ("" when the site was estimated).

suppressPackageStartupMessages({
  library(arrow)
  library(jsonlite)
})

# ----------------------------------------------------------------- arguments
parse_args <- function(argv) {
  opts <- list(bundle = NULL, method = NULL, files = NULL, out = NULL,
               min_n = 20L, maxiter = 1000L, tol = 1e-7, threads = 1L)
  i <- 1L
  while (i <= length(argv)) {
    key <- argv[[i]]
    val <- if (i < length(argv)) argv[[i + 1L]] else NA_character_
    switch(key,
      "--bundle"  = { opts$bundle  <- val; i <- i + 2L },
      "--method"  = { opts$method  <- val; i <- i + 2L },
      "--files"   = { opts$files   <- as.integer(strsplit(val, ",")[[1]]); i <- i + 2L },
      "--out"     = { opts$out     <- val; i <- i + 2L },
      "--min-n"   = { opts$min_n   <- as.integer(val); i <- i + 2L },
      "--maxiter" = { opts$maxiter <- as.integer(val); i <- i + 2L },
      "--tol"     = { opts$tol     <- as.numeric(val); i <- i + 2L },
      "--threads" = { opts$threads <- as.integer(val); i <- i + 2L },
      "--help"    = { cat(readLines(sub("--file=", "", grep("--file=", commandArgs(), value = TRUE))[1:12]), sep = "\n"); quit(status = 0) },
      stop("unknown argument: ", key)
    )
  }
  if (is.null(opts$bundle) || is.null(opts$method)) {
    stop("--bundle and --method are required")
  }
  opts
}

METHODS <- c("pl", "al", "pg", "ag", "phase_pg", "phase_ag")

opts <- parse_args(commandArgs(trailingOnly = TRUE))
if (!opts$method %in% METHODS) stop("--method must be one of ", paste(METHODS, collapse = ", "))

manifest <- fromJSON(file.path(opts$bundle, "manifest.json"), simplifyVector = TRUE)
if (!identical(as.integer(manifest$format_version), 1L)) {
  stop("unsupported bundle format_version ", manifest$format_version)
}
K <- as.integer(manifest$K)
n <- as.integer(manifest$layout$n_samples)
W <- as.integer(manifest$layout$windows_per_file)
ancestries <- manifest$ancestries
has_post <- isTRUE(manifest$has_posterior)
has_gl <- !is.null(manifest$gl_source) && !is.na(manifest$gl_source)

needs_post <- opts$method %in% c("pl", "pg", "phase_pg")
needs_gl <- opts$method %in% c("pl", "al")
if (needs_post && !has_post) {
  stop("this bundle carries hard labels only (no posteriors); use one of: al, ag, phase_ag")
}
if (needs_gl && !has_gl) {
  stop("this bundle has no genotype likelihoods (no PL/GL in the VCF); use one of: ",
       if (has_post) "pg, ag, phase_pg, phase_ag" else "ag, phase_ag")
}
if (opts$method != "phase_ag") {
  suppressPackageStartupMessages({
    library(SQUAREM)   # see the note above: LoGicAl does not import it
    library(LoGicAl)
  })
}

files_meta <- manifest$layout$files
if (is.data.frame(files_meta)) {
  files_meta <- split(files_meta, seq_len(nrow(files_meta)))
}
file_ids <- vapply(files_meta, function(f) as.integer(f$k), integer(1))
if (!is.null(opts$files)) {
  files_meta <- files_meta[file_ids %in% opts$files]
}
if (length(files_meta) == 0L) stop("no files selected")

# ----------------------------------------------------------------- helpers
pcols <- if (has_post) paste0("p_", ancestries) else character(0)

read_posteriors <- function(window_min, window_max) {
  # labels (n_win x n x 2) and, when present, probabilities (n_win x n x 2 x K)
  # for the posterior files covering [window_min, window_max]
  j0 <- window_min %/% W
  j1 <- window_max %/% W
  cols <- c("label", pcols)
  tabs <- lapply(j0:j1, function(j) {
    as.data.frame(read_parquet(file.path(opts$bundle, sprintf("posterior-%d.parquet", j)),
                               col_select = all_of(cols)))
  })
  df <- do.call(rbind, tabs)
  first <- j0 * W
  n_win <- nrow(df) %/% (n * 2L)
  # rows are (window, sample, hap) dense: aperm to [hap, sample, window] then index
  label <- array(as.integer(df$label), dim = c(2L, n, n_win))
  probs <- NULL
  if (has_post) {
    probs <- array(unlist(df[pcols], use.names = FALSE), dim = c(2L, n, n_win, K))
  }
  list(first = first, label = label, probs = probs)
}

closed_form <- function(z, h, K) {
  # z, h: vectors over haplotypes (both haplotypes of every individual)
  vapply(seq_len(K), function(k) {
    sel <- z == k
    if (!any(sel)) return(NA_real_)
    mean(h[sel] == 1)
  }, numeric(1))
}

estimate_site <- function(z1, z2, pi1, pi2, g, h1, h2, gl) {
  method <- opts$method
  keep <- switch(method,
    "pl"       = rowSums(is.na(gl)) == 0,
    "al"       = z1 > 0 & z2 > 0 & rowSums(is.na(gl)) == 0,
    "pg"       = g >= 0,
    "ag"       = z1 > 0 & z2 > 0 & g >= 0,
    "phase_pg" = h1 >= 0 & h2 >= 0,
    "phase_ag" = z1 > 0 & z2 > 0 & h1 >= 0 & h2 >= 0
  )
  if (method %in% c("pl", "pg", "phase_pg")) {
    # a haplotype with no posterior mass contributes nothing: drop it
    keep <- keep & rowSums(pi1) > 0 & rowSums(pi2) > 0
  }
  if (method == "phase_ag") {
    use1 <- z1 > 0 & h1 >= 0
    use2 <- z2 > 0 & h2 >= 0
    n_used <- sum(use1 | use2)
    n_hap <- sum(use1) + sum(use2)
    if (n_used < opts$min_n) {
      return(list(est = rep(NA_real_, K), n_used = n_used, n_hap = n_hap, error = "below min-n"))
    }
    est <- closed_form(c(z1[use1], z2[use2]), c(h1[use1], h2[use2]), K)
    return(list(est = est, n_used = n_used, n_hap = n_hap, error = ""))
  }
  n_used <- sum(keep)
  if (n_used < opts$min_n) {
    return(list(est = rep(NA_real_, K), n_used = n_used, n_hap = 2L * n_used, error = "below min-n"))
  }
  res <- tryCatch({
    est <- LoGicAl::run_logical(
      pi1 = pi1[keep, , drop = FALSE], pi2 = pi2[keep, , drop = FALSE],
      z1 = z1[keep], z2 = z2[keep],
      gl = if (is.null(gl)) matrix(1, n_used, 4L) else gl[keep, , drop = FALSE],
      g = g[keep], h1 = h1[keep], h2 = h2[keep],
      method = method, par0 = rep(1 / K, K),
      maxiter = opts$maxiter, tol = opts$tol
    )
    est <- as.numeric(est$par)
    list(est = pmin(pmax(est, 0), 1), n_used = n_used, n_hap = 2L * n_used, error = "")
  }, error = function(e) {
    list(est = rep(NA_real_, K), n_used = n_used, n_hap = 2L * n_used, error = conditionMessage(e))
  })
  res
}

run_file <- function(f) {
  k <- as.integer(f$k)
  sites <- as.data.frame(read_parquet(file.path(opts$bundle, sprintf("sites-%d.parquet", k))))
  geno <- as.data.frame(read_parquet(file.path(opts$bundle, sprintf("genotypes-%d.parquet", k))))
  m <- nrow(sites)
  stopifnot(nrow(geno) == m * n)
  post <- read_posteriors(as.integer(f$window_min), as.integer(f$window_max))
  gl_all <- if (has_gl) as.matrix(geno[, c("gl_00", "gl_10", "gl_01", "gl_11")]) else NULL

  one <- function(v) {
    rows <- ((v - 1L) * n + 1L):(v * n)
    w <- sites$window[v] - post$first + 1L
    z1 <- post$label[1L, , w]
    z2 <- post$label[2L, , w]
    if (has_post) {
      pi1 <- matrix(post$probs[1L, , w, ], nrow = n, ncol = K)
      pi2 <- matrix(post$probs[2L, , w, ], nrow = n, ncol = K)
    } else {
      pi1 <- pi2 <- matrix(0, n, K)
    }
    r <- estimate_site(z1, z2, pi1, pi2, geno$g[rows], geno$h1[rows], geno$h2[rows],
                       if (has_gl) gl_all[rows, , drop = FALSE] else NULL)
    c(list(n_used = r$n_used, n_hap = r$n_hap, error = r$error), as.list(r$est))
  }
  res <- if (opts$threads > 1L) {
    parallel::mclapply(seq_len(m), one, mc.cores = opts$threads)
  } else {
    lapply(seq_len(m), one)
  }
  est <- do.call(rbind, lapply(res, function(r) unlist(r[-(1:3)])))
  colnames(est) <- paste0(ancestries, "_freq")
  n_used <- vapply(res, function(r) as.integer(r$n_used), integer(1))
  data.frame(
    chrom = manifest$chromosome, pos = sites$pos, id = sites$id, ref = sites$ref, alt = sites$alt,
    window = sites$window, distance = sites$distance, method = opts$method,
    n_used = n_used, n_dropped = n - n_used,
    n_haplotypes = vapply(res, function(r) as.integer(r$n_hap), integer(1)),
    est,
    error = vapply(res, function(r) as.character(r$error), character(1)),
    stringsAsFactors = FALSE
  )
}

out <- do.call(rbind, lapply(files_meta, run_file))
if (is.null(opts$out)) {
  write.table(out, file = stdout(), sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
} else {
  write.table(out, file = opts$out, sep = "\t", quote = FALSE, row.names = FALSE, na = "NA")
  message(sprintf("%d sites x %s -> %s", nrow(out), opts$method, opts$out))
}
