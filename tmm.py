import numpy as np
import pandas as pd

# Written by GPT 5

def tmm_factors(
    counts: pd.DataFrame,
    ref_col: str | None = None,
    logratio_trim: float = 0.30,
    sum_trim: float = 0.05,
    weight=True,
    min_count: int = 1,
):
    """
    Compute TMM (Trimmed Mean of M-values) normalization factors for a counts matrix.

    Parameters
    ----------
    counts : DataFrame
        Genes (rows) x Samples (columns) raw counts (non-negative integers).
    ref_col : str or None
        Column name to use as reference sample. If None, pick sample with median library size.
    logratio_trim : float
        Proportion to trim off each tail of M (log-ratio) distribution (edgeR default ~0.30).
    sum_trim : float
        Proportion to trim off each tail of A (average log expression) distribution (edgeR default ~0.05).
    weight : bool
        If True, use inverse-variance weights ~ 1/(1/c_i + 1/c_ref) (good approximation to edgeR).
    min_count : int
        Exclude genes with count < min_count in either sample.

    Returns
    -------
    Series
        TMM normalization factors (indexed by sample). Effective lib size = lib_size * factor.
    """
    if not isinstance(counts, pd.DataFrame):
        raise TypeError("counts must be a pandas.DataFrame (genes x samples)")

    # library sizes
    lib_sizes = counts.sum(axis=0).astype(float)

    # pick reference
    if ref_col is None:
        ref_col = lib_sizes.sort_values().index[len(lib_sizes)//2]
    if ref_col not in counts.columns:
        raise ValueError(f"ref_col {ref_col} not in counts columns")

    ref = counts[ref_col].astype(float).values
    N_ref = lib_sizes[ref_col]

    # precompute masks for zero/low counts against the reference once (gene-level)
    # We'll also avoid division by zero by masking zeros.
    def _tmm_to_ref(sample_col):
        y = counts[sample_col].astype(float).values
        N_y = lib_sizes[sample_col]

        # keep genes sufficiently expressed in both
        keep = (y >= min_count) & (ref >= min_count)
        if keep.sum() == 0:
            return np.nan  # degenerate (all filtered)

        yk = y[keep]; rk = ref[keep]

        # CPM-ish (proportional) rates
        py = yk / N_y
        pr = rk / N_ref

        # M and A (log2)
        with np.errstate(divide='ignore'):
            M = np.log2(py) - np.log2(pr)
            A = 0.5 * (np.log2(py) + np.log2(pr))

        # Finite only
        finite = np.isfinite(M) & np.isfinite(A)
        if finite.sum() == 0:
            return np.nan
        M = M[finite]; A = A[finite]
        yk = yk[finite]; rk = rk[finite]

        # Trimming by M and A
        n = M.size
        m_lo = int(np.floor(logratio_trim/2 * n))
        m_hi = n - m_lo
        a_lo = int(np.floor(sum_trim/2 * n))
        a_hi = n - a_lo

        order_M = np.argsort(M)
        order_A = np.argsort(A)
        keep_M = np.zeros(n, dtype=bool); keep_M[order_M[m_lo:m_hi]] = True
        keep_A = np.zeros(n, dtype=bool); keep_A[order_A[a_lo:a_hi]] = True
        keep_both = keep_M & keep_A

        if keep_both.sum() == 0:
            return np.nan

        Mk = M[keep_both]
        if weight:
            # Approx inverse-variance weights (Robinson & Oshlack 2010 style):
            # var(M_g) ≈ (1/yk + 1/rk) on the count scale (good practical proxy)
            wk = 1.0 / (1.0/yk[keep_both] + 1.0/rk[keep_both])
            mbar = np.sum(wk * Mk) / np.sum(wk)
        else:
            mbar = np.mean(Mk)

        # scaling factor for sample relative to ref; edgeR uses factor = 2^{-mbar}
        return float(2.0 ** (mbar))

    factors = {}
    for col in counts.columns:
        if col == ref_col:
            factors[col] = 1.0  # reference gets factor 1 by convention
        else:
            factors[col] = _tmm_to_ref(col)

    # Normalize factors to have geometric mean = 1 (edgeR behavior)
    f = pd.Series(factors)
    gm = np.exp(np.nanmean(np.log(f.dropna())))
    f = f / gm
    return f


def cpm(counts: pd.DataFrame, tmm: pd.Series) -> pd.DataFrame:
    """
    Compute CPM using TMM-adjusted effective library sizes.
    CPM = 1e6 * (counts) / (lib_size * tmm_factor)
    """
    lib_sizes = counts.sum(axis=0).astype(float)
    eff_lib = lib_sizes * tmm[counts.columns].values
    return counts.div(eff_lib, axis=1) * 1e6

def tmm_normalize(mtx, pseudocount=1):
    factors = tmm_factors(mtx)
    cpm_tmm = cpm(mtx, factors)
    return np.log2(cpm_tmm + pseudocount)

def rsem_normalize(df, target=None):
    uq = df.quantile(0.75, axis=1)
    if target is None:
        target = uq.median()
    sf = uq / target
    df_uq = df.divide(sf, axis=0)
    df_log = np.log2(df_uq + 1.0)
    return df_log

from adjustText import adjust_text
from scipy.stats import linregress
import matplotlib.pyplot as plt
import seaborn as sns

# helper scatterplot function
def scatterplot(df, x_name, y_name, ax, title='', legend=True, plot_texts=False,
                legend_loc=(1.02, 0.01), text_fontsize=10, adjust_text_expand=(1.25, 1.5)):
    # Extract 'sample' from index
    df['sample'] = df.index.str.split('_').str[0]
    sc = sns.scatterplot(
        data=df,
        x=x_name, 
        y=y_name, 
        hue='sample',
        palette='tab20',
        s=80,
        ax=ax
    )
    
    if plot_texts:
        texts = []
        for cluster, row in df.iterrows():
            texts.append(ax.text(row[x_name], row[y_name], cluster, fontsize=text_fontsize))
        
        # Pass scatter points to adjust_text so labels avoid them
        adjust_text(
            texts,
            ax=ax,
            add_objects=sc.collections,  # these are the scatter points
            arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
            expand=adjust_text_expand,
        ) 

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    if legend:
        ax.legend(title='Sample', loc = legend_loc)
    else:
        ax.legend([])

def line_projection_score(df, xcol, ycol, ax=None, line_kwargs=None, proj_kwargs=None):
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()

    # 1) Fit y = a + b x
    lr = linregress(x, y)
    a, b = lr.intercept, lr.slope

    # 2) Plot line of best fit (optional)
    if ax is not None:
        xx = np.linspace(x.min(), x.max(), 200)
        ax.plot(xx, a + b*xx, **({"lw": 2, "alpha": 0.9} | (line_kwargs or {})))

    # 3) Project each point p onto the line defined by point p_ref and direction v=[1, b]
    v = np.array([1.0, b])
    u = v / np.linalg.norm(v)                  # unit direction along the line
    x0 = x.mean()
    p_ref = np.array([x0, a + b*x0])           # any point on the line works; use mean-x anchor
    P = np.column_stack([x, y])
    t = (P - p_ref) @ u                        # signed distance along the line
    P_proj = p_ref + np.outer(t, u)            # projected points on the line

    # 4) Normalize to 0–1
    tmin, tmax = t.min(), t.max()
    scores = np.zeros_like(t) if tmax == tmin else (t - tmin) / (tmax - tmin)

    # Optionally draw projected points
    if ax is not None and proj_kwargs is not None:
        ax.scatter(P_proj[:,0], P_proj[:,1], **proj_kwargs)

    return lr, scores, P_proj
