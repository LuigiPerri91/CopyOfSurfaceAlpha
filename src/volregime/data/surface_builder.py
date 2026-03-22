import numpy as np 
import pandas as pd
import logging
from scipy.interpolate import griddata
from dotenv import load_dotenv
import yaml

load_dotenv()

with open('../configs/default.yaml','r') as f:
    default = yaml.safe_load(f)

with open('../configs/data.yaml','r') as f:
    data = yaml.safe_load(f)

logger = logging.getLogger(__name__)

def build_surface(option_rows, surface_config, is_gap_filled=False, gap_days=0):
    """
    Convert a single (date, symbol) batch of option rows into a fixed-grid surface tensor.

    Args:
        option_rows: DataFrame for one (date, symbol) with columns:
            [iv, bid, ask, delta, vega, moneyness, tau, call_put]
            moneyness = strike/spot (already computed)
            tau = (expiration - date)/365 (already computed)
        surface_config: the data.yaml surface section
        is_gap_filled: whether this date has no real data (carried forward)
        gap_days: how many days since last real observation

    Returns:
        surface_tensor: numpy array of shape (num_channels, n_maturity_bins, n_moneyness_bins)
    """

    n_m = surface_config['n_moneyness_bins'] # width =20
    n_t = surface_config['n_maturity_bins'] # height =12
    maturity_buckets = surface_config['maturity_buckets'] 
    put_call_mode = surface_config.get('put_call_mode', 'otm_only')
    interp_method = surface_config.get("interpolation", "linear")

    # define grid edges
    # moneyness axis: evenly spaced in log-moneyness from ln(0.80) to ln(1.20)
    log_m_min = np.log(data['moneyness']['min'])
    log_m_max = np.log(data['moneyness']['max'])
    moneyness_edges = np.linspace(log_m_min, log_m_max, n_m + 1)
    moneyness_centers = (moneyness_edges[:-1] + moneyness_edges[1:]) / 2

    # maturity axis: use bucket boundaries from config, convert days -> years
    maturity_centers_years = [d /365 for d in maturity_buckets]

    # filter by put_call_mode
    rows = option_rows.copy()
    if put_call_mode == 'otm_only':
        # keep OTM puts (moneyness < 1) and OTM calls (moneyness > 1)
        rows = rows[
            ((rows['call_put'] == 'P') & (rows['moneyness'] < 1.0)) |
            ((rows['call_put'] == 'C') & (rows['moneyness'] >= 1.0))
        ]

    # compute log-moneyness for each row
    rows['log_m'] = np.log(rows['moneyness'])

    # initialize grids
    num_channels = sum(data['surface']['channels'].values()) # iv, spread_norm, mask, staleness, delta, vega
    grid = np.zeros((num_channels, n_t, n_m), dtype=np.float32)

    # assign each option to nearest grid cell
    for _, row in rows.iterrows():
        # find nearest moneynees bin
        m_idx = np.argmin(np.abs(np.array(moneyness_centers) - row['log_m']))

        # find nearest maturity bin
        t_idx = np.argmin(np.abs(np.array(maturity_centers_years) - row['tau']))

        # if multiple options map to same cell, average them
        # (using a count array to track)
        midpoint = (row['ask'] + row['bid']) / 2
        spread_norm = (row['ask'] - row['bid']) / midpoint if midpoint > 0 else 0

        # accumulate (will divide by count later)
        grid[0, t_idx, m_idx] += row['iv']
        grid[1, t_idx, m_idx] += spread_norm
        grid[2, t_idx, m_idx] += 1
        grid[4, t_idx, m_idx] += row['delta']
        grid[5, t_idx, m_idx] += row['vega']

    counts = grid[2].copy()
    counts_safe = np.where(counts>0, counts, 1)

    grid[0] /= counts_safe
    grid[1] /= counts_safe
    grid[4] /= counts_safe
    grid[5] /= counts_safe

    # mask 
    # mask = 1 where at least one observation, 0 otherwise
    mask = (counts>0).astype(np.float32)
    grid[2] = mask

    # staleness
    if is_gap_filled:
        grid[3] = float(gap_days)
    else:
        grid[3] = 0.0

    # within date interpolation for empty cells
    if interp_method != 'none' and mask.sum() >= 4:
        # only interpolate channels 0,1,4,5 (not mask or staleness)
        observed_coords = np.argwhere(mask>0) # (N,2) array of [t_idx, m_idx]
        target_coords = np.argwhere(mask==0) # (N,2) array of [t_idx, m_idx]
        if len(target_coords) > 0:
            for c in [0,1,4,5]:
                observed_values = grid[c][mask>0]
                interpolated = griddata(observed_coords, observed_values, target_coords, method=interp_method, fill_value=0.0)
                for k, (ti, mi) in enumerate(target_coords):
                    # only fill if interpolation succeeded (not NaN)
                    if not np.isnan(interpolated[k]):
                        grid[c, ti, mi] = interpolated[k]
    return grid # shape : (num_channels, n_t, n_m)