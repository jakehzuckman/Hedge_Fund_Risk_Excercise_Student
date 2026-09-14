import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

def forward_stepwise(y, X_pool, min_adj_r2_gain=0.01, verbose=True):
    """
    Forward stepwise regression (adjusted-R-squared criterion).
    
    At each step, try adding every remaining factor to the current model.
    Keep the factor whose addition yields the highest adjusted R-squared,
    but only if the MARGINAL improvement in adjusted R-squared exceeds
    min_adj_r2_gain.  Repeat until no factor clears the hurdle.
    
    Returns: final OLS result, DataFrame of steps taken
    """
    import statsmodels.api as sm
    
    selected = []
    remaining = list(X_pool.columns)
    steps = []
    
    # Baseline: intercept-only model
    X_base = sm.add_constant(pd.DataFrame(index=y.index))
    current_adj_r2 = sm.OLS(y, X_base).fit().rsquared_adj
    
    for step_num in range(1, len(remaining) + 1):
        best_adj_r2 = -np.inf
        best_factor = None
        best_res = None
        
        for factor in remaining:
            trial = selected + [factor]
            X_trial = sm.add_constant(X_pool[trial])
            res = sm.OLS(y, X_trial).fit()
            if res.rsquared_adj > best_adj_r2:
                best_adj_r2 = res.rsquared_adj
                best_factor = factor
                best_res = res
        
        gain = best_adj_r2 - current_adj_r2
        
        if gain < min_adj_r2_gain:
            if verbose:
                print(f'  Step {step_num}: Best candidate ({best_factor}) improves '
                      f'adj-R-sq by only {gain:+.4f} (< {min_adj_r2_gain}). Stopping.')
            break
        
        selected.append(best_factor)
        remaining.remove(best_factor)
        current_adj_r2 = best_adj_r2
        
        steps.append({
            'Step': step_num,
            'Factor Added': best_factor,
            'Beta': f'{best_res.params[best_factor]:.4f}',
            't-stat': f'{best_res.tvalues[best_factor]:.2f}',
            'p-value': f'{best_res.pvalues[best_factor]:.4f}',
            'Adj-R-sq': f'{best_adj_r2:.4f}',
            'Marginal Gain': f'{gain:+.4f}',
        })
        
        if verbose:
            print(f'  Step {step_num}: + {best_factor:12s}  '
                  f'B={best_res.params[best_factor]:+7.4f}  '
                  f't={best_res.tvalues[best_factor]:+6.2f}  '
                  f'p={best_res.pvalues[best_factor]:.4f}  '
                  f'Adj-R-sq={best_adj_r2:.4f}  '
                  f'(+{gain:.4f})')
    
    if selected:
        X_final = sm.add_constant(X_pool[selected])
        res_final = sm.OLS(y, X_final).fit()
    else:
        X_final = sm.add_constant(pd.DataFrame(index=y.index))
        res_final = sm.OLS(y, X_final).fit()
    
    # Print final model betas
    if verbose and selected:
        print(f'\n  Final Model Betas:')
        for f in selected:
            print(f'    {f:12s}  B = {res_final.params[f]:+.4f}')
        print(f'    {"Alpha":12s}  B = {res_final.params["const"]:+.4f}  '
              f'(ann: {res_final.params["const"]*12:+.2%})')
    
    return res_final, pd.DataFrame(steps)

print('forward_stepwise() defined.')

# Assume these are already defined in your notebook:
# - funds: DataFrame of fund returns
# - factors: DataFrame of factor returns
# - WINDOW: rolling window (e.g., 12 months)
# - VISIBLE_FACTORS: list of factor names to display
# - dates: date index
# - N: total number of observations
# - all_results: dictionary of full regression results per fund

def calculate_rolling_betas_with_ci(funds, factors, WINDOW, VISIBLE_FACTORS, dates, N, all_results, ci=0.95):
    """
    Calculate rolling betas and confidence intervals.
    
    Returns:
        dict: {fund_name: {'betas': DataFrame, 'ci_lower': DataFrame, 'ci_upper': DataFrame}}
    """
    results = {}
    z_score = stats.norm.ppf((1 + ci) / 2)  # 1.96 for 95% CI
    
    for fund_name in funds.columns:
        roll_betas = []
        roll_ci_lower = []
        roll_ci_upper = []
        roll_dates = []
        roll_se = []  # standard errors for each rolling window
        
        for end in range(WINDOW, N):
            start = end - WINDOW
            X_roll = sm.add_constant(factors[VISIBLE_FACTORS].iloc[start:end])
            y_roll = funds[fund_name].iloc[start:end]
            res_roll = sm.OLS(y_roll, X_roll).fit()
            
            params = res_roll.params.drop('const')
            std_errs = res_roll.bse.drop('const')
            
            roll_betas.append(params)
            roll_se.append(std_errs)
            roll_dates.append(dates[end])
            
            # Calculate CI: beta ± z * SE
            roll_ci_lower.append(params - z_score * std_errs)
            roll_ci_upper.append(params + z_score * std_errs)
        
        results[fund_name] = {
            'betas': pd.DataFrame(roll_betas, index=roll_dates),
            'ci_lower': pd.DataFrame(roll_ci_lower, index=roll_dates),
            'ci_upper': pd.DataFrame(roll_ci_upper, index=roll_dates),
        }
    
    return results


def plot_rolling_betas_with_ci(funds, factors, WINDOW, VISIBLE_FACTORS, dates, N, all_results, ci=0.95, n_factors=3):
    """
    Plot rolling betas with confidence interval bands.
    
    Args:
        funds: DataFrame of fund returns (columns = fund names)
        factors: DataFrame of factor returns
        WINDOW: rolling window size (e.g., 12 for 12-month)
        VISIBLE_FACTORS: list of factor names to include in regression
        dates: date index
        N: total number of observations
        all_results: dict of full-sample OLS results per fund (for ranking factors)
        ci: confidence level (default 0.95 for 95% CI)
        n_factors: number of top factors to display per fund (default 3)
            - Factors ranked by absolute beta magnitude from full-sample OLS
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 32), sharex=True)
    z_score = stats.norm.ppf((1 + ci) / 2)  # 1.96 for 95% CI
    
    for ax, fund_name in zip(axes.flatten(), funds.columns):
        roll_betas = []
        roll_ci_lower = []
        roll_ci_upper = []
        roll_dates = []
        
        # Compute rolling regressions
        for end in range(WINDOW, N):
            start = end - WINDOW
            X_roll = sm.add_constant(factors[VISIBLE_FACTORS].iloc[start:end])
            y_roll = funds[fund_name].iloc[start:end]
            res_roll = sm.OLS(y_roll, X_roll).fit()
            
            params = res_roll.params.drop('const')
            std_errs = res_roll.bse.drop('const')
            
            roll_betas.append(params)
            roll_ci_lower.append(params - z_score * std_errs)
            roll_ci_upper.append(params + z_score * std_errs)
            roll_dates.append(dates[end])
        
        # Create DataFrames for plotting
        roll_df = pd.DataFrame(roll_betas, index=roll_dates)
        roll_ci_lower_df = pd.DataFrame(roll_ci_lower, index=roll_dates)
        roll_ci_upper_df = pd.DataFrame(roll_ci_upper, index=roll_dates)
        
        # Select top N factors to display (by absolute beta magnitude from full sample OLS)
        full_sample_betas = all_results[fund_name].params.drop('const').abs().sort_values(ascending=False)
        top = full_sample_betas.nlargest(n_factors).index.tolist()
        # Filter to only include factors that appear in rolling regression results
        top = [f for f in top if f in roll_df.columns]
        # Fallback: if no matches, take first n_factors from rolling results
        if not top:
            top = roll_df.columns[:n_factors].tolist()
        
        # Plot each factor with confidence interval band
        for factor in top:
            ax.plot(roll_df.index, roll_df[factor], linewidth=1.5, label=factor, alpha=0.8)
            ax.fill_between(
                roll_df.index,
                roll_ci_lower_df[factor],
                roll_ci_upper_df[factor],
                alpha=0.15  # Light shading for CI
            )
        
        # Formatting
        ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
        ax.set_title(f'{fund_name} (shaded = 95% CI)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Beta (12m Rolling)')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.2)
    
    fig.suptitle(f'Rolling {WINDOW}-Month Betas with Confidence Intervals: Exposure Stability',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

def compute_all_results(funds, factors):
    """
    Compute full-sample OLS results for all funds.
    
    Args:
        funds: DataFrame of fund returns (columns = fund names)
        factors: DataFrame of factor returns
    
    Returns:
        dict: {fund_name: OLS result object}
    """
    all_results = {}
    for fund_name in funds.columns:
        X = sm.add_constant(factors)
        y = funds[fund_name]
        result = sm.OLS(y, X).fit()
        all_results[fund_name] = result
    
    return all_results
