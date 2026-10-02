import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .config import BINS

def make_plots(results, path, progress=lambda x:x):
    path.mkdir(exist_ok=True)
    groups=list(results.groupby(['source','signal_name','horizon'],observed=True))
    for (source,signal,horizon),group in progress(groups):
        fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
        for ax,col,title in zip(axes[:2],['mean_return','mean_excess_vs_unconditional'],['Mean return','Excess vs eligible benchmark']):
            table=group.pivot(index='rolling_window',columns='percentile_bin',values=col).reindex(columns=BINS)
            data=table.to_numpy(dtype=float)
            finite=np.abs(data[np.isfinite(data)]); limit=finite.max() if len(finite) else 1
            limit=max(limit,1e-8)
            im=ax.imshow(data,aspect='auto',cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax.set_xticks(range(7),BINS,rotation=45); ax.set_yticks(range(len(table)),table.index)
            ax.set_title(title); ax.set_ylabel('Rolling window'); fig.colorbar(im,ax=ax)
        for window,g in group.groupby('rolling_window'):
            curve=g.set_index('percentile_bin').reindex(BINS).mean_excess_vs_unconditional
            axes[2].plot(range(7),curve,label=str(window),marker='o')
        axes[2].set_xticks(range(7),BINS,rotation=45); axes[2].axhline(0,color='gray',lw=.7)
        axes[2].set_title('Percentile response: excess'); axes[2].legend(fontsize=7)
        fig.suptitle(f'{source} {signal} C{horizon} — descriptive')
        fig.savefig(path/f'{source}_{signal}_C{horizon}.png',dpi=120)
        plt.close(fig)
