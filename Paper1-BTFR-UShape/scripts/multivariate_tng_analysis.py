import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

df = pd.read_csv('../data/tng_galaxies.csv')
d = pd.DataFrame({'resid': df['btfr_residual']})
stats = {}
for name, src in [('env', df['log_env']), ('fgas', df['gas_fraction']), ('lmstar', np.log10(df['stellar_mass']))]:
    stats[name] = (src.mean(), src.std())
    d[name] = (src - src.mean())/src.std()
d['env2'] = d['env']**2

m0 = smf.ols('resid ~ env + env2', d).fit(cov_type='HC3')
m1 = smf.ols('resid ~ env + env2 + fgas + lmstar', d).fit(cov_type='HC3')
m2 = smf.ols('resid ~ env + env2 + fgas + lmstar + env:fgas + env:lmstar + env2:fgas', d).fit(cov_type='HC3')

fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
plt.rcParams.update({'font.size': 13})

# Panel A: env2 coefficient across nested models
ax = axes[0]
labels = ['M0\n(env only)', 'M1\n(+ controls)', 'M2\n(+ interactions)']
coefs = [m.params['env2'] for m in (m0, m1, m2)]
errs  = [1.96*m.bse['env2'] for m in (m0, m1, m2)]
ax.errorbar(range(3), coefs, yerr=errs, fmt='o', ms=9, capsize=5, lw=2, color='#1a5fb4')
ax.axhline(0, color='crimson', ls='--', lw=1.2)
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=12)
ax.set_ylabel('Quadratic environmental coefficient (std.)', fontsize=13)
ax.set_title('(a) Curvature survives controls', fontsize=14)
for i,(c,m) in enumerate(zip(coefs,(m0,m1,m2))):
    ax.annotate(f"{c/m.bse['env2']:+.1f}Ïƒ", (i, c), textcoords='offset points', xytext=(12,8), fontsize=12)

# Panel B: marginal environmental slope as function of fgas (continuous inversion)
ax = axes[1]
fg_grid = np.linspace(d['fgas'].quantile(0.02), d['fgas'].quantile(0.98), 100)
b, cov = m2.params, m2.cov_params()
slope = b['env'] + b['env:fgas']*fg_grid
se = np.sqrt(cov.loc['env','env'] + fg_grid**2*cov.loc['env:fgas','env:fgas'] + 2*fg_grid*cov.loc['env','env:fgas'])
fg_phys = fg_grid*stats['fgas'][1] + stats['fgas'][0]
ax.plot(fg_phys, slope, color='#1a5fb4', lw=2.5)
ax.fill_between(fg_phys, slope-1.96*se, slope+1.96*se, alpha=0.25, color='#1a5fb4')
ax.axhline(0, color='crimson', ls='--', lw=1.2)
x0 = -b['env']/b['env:fgas']*stats['fgas'][1] + stats['fgas'][0]
ax.axvline(x0, color='gray', ls=':', lw=1.5)
ax.annotate(f'sign inversion\nat f_gas â‰ˆ {x0:.2f}', (x0, slope.max()*0.55), fontsize=12, ha='left', xytext=(x0+0.05, slope.max()*0.55))
ax.set_xlabel('Gas fraction', fontsize=13)
ax.set_ylabel('Marginal env. slope  d(resid)/d(env)', fontsize=13)
ax.set_title('(b) Continuous sign inversion with $f_{gas}$', fontsize=14)

# Panel C: partial residuals (controls removed) binned by env, split by fgas
ax = axes[2]
ctrl = smf.ols('resid ~ fgas + lmstar', d).fit()
d['presid'] = ctrl.resid
for mask, lab, col in [(d['fgas']>=d['fgas'].quantile(0.75),'gas-rich (top quartile)','#26a269'),
                        (d['fgas']<=d['fgas'].quantile(0.25),'star-dominated (quenched, f_gas=0)','#c01c28')]:
    sub = d[mask]
    bins = np.quantile(sub['env'], np.linspace(0.02,0.98,12))
    ctr = 0.5*(bins[1:]+bins[:-1])
    means = [sub['presid'][(sub['env']>=bins[i])&(sub['env']<bins[i+1])].mean() for i in range(11)]
    sems  = [sub['presid'][(sub['env']>=bins[i])&(sub['env']<bins[i+1])].sem() for i in range(11)]
    env_phys = ctr*stats['env'][1] + stats['env'][0]
    ax.errorbar(env_phys, means, yerr=np.array(sems)*1.96, fmt='o-', ms=5, capsize=3, label=lab, color=col, lw=1.8)
ax.axhline(0, color='gray', ls='--', lw=1)
ax.set_xlabel(r'$\log\,\rho_{env}$', fontsize=13)
ax.set_ylabel('Partial BTFR residual (controls removed)', fontsize=13)
ax.set_title('(c) Opposite environmental response', fontsize=14)
ax.legend(fontsize=11, loc='upper left')

plt.tight_layout()
plt.savefig('./multivariate_tng_analysis.png', dpi=200, bbox_inches='tight')
print('Figure saved.')

# Save results table
with open('./multivariate_results.txt','w') as f:
    for lab, m in [('M0: env only',m0),('M1: + controls',m1),('M2: + interactions',m2)]:
        f.write('='*78+f'\n{lab} | N={int(m.nobs)} R2={m.rsquared:.4f} AIC={m.aic:.0f} BIC={m.bic:.0f}\n')
        f.write(m.summary().tables[1].as_text()+'\n\n')
print('Results table saved.')

