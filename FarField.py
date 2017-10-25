


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import cartopy.crs as ccrs


def AxisSetUp(ax,ymin,ymax,xmin,xmax,text):
    # plot format
    for label in ax.get_ymajorticklabels():
        label.set_rotation(90)
        label.set_verticalalignment("center")
        label.set_horizontalalignment("center")


    ax.grid(True,zorder=1)
    ax.set_facecolor('#DAE3FE')

    ax.set_ylim(ymin,ymax)
    ax.set_xlim(xmin,xmax)

    # create tick


    # ax.xaxis_set_major_ticks()
    # ax.yaxis_set_major_ticks()

    MajorTicks=zip(ax.xaxis.get_major_ticks(),
                   ax.yaxis.get_major_ticks())

    # xticklabels inside
    for xtick,ytick in MajorTicks:
        xtick.set_pad(-13)
        ytick.set_pad(-13)

    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.tick_params(labelsize=6)

    YText=ymax*0.85+ymin*0.15
    XText=xmax*0.85+xmin*0.15

    ax.text(XText, YText,
             text,
             fontsize=9,
             bbox={
                 'facecolor':'white',
                 'alpha':1,
                 'pad':10}
             )

if __name__=='__main__':


    kind='i:\\OPS\\005-MOG\\15-001_UO-BS\\Operacao\\Sol41_Efl_PCP-2\\Oper\
acao\\Tratamento_e_Analise\\Sig\\D_Shapes\\AGPROD_inv_prob'


    fnames = glob('{}*.shp'.format(kind))

    xmin=None
    xmax=None
    ymin=None
    ymax=None

    xmax=-40.38
    xmin=-40.43

    dx=xmax-xmin
    dy=dx/1.59
    ymin=-22.25
    ymax=ymin+dy

    for fname in fnames:

        print(fname)

        fig,(ax1,ax2) = plt.subplots(nrows=2,
                                    ncols=1,
                                    figsize=(8.27, 11.69)
                                    )
        fig.tight_layout()

        AxisSetUp(ax1,ymin,ymax,xmin,xmax,'Diluição')
        AxisSetUp(ax2,ymin,ymax,xmin,xmax,'Probabilidade')

        gdf = gpd.read_file(fname)

        gdf.plot(ax=ax1,
                 column='prob_min',
                 cmap='OrRd',
                 zorder=10
                 )

        fig.savefig('teste.png',dpi=300)
