
from glob import glob
from numpy import arange,ceil,abs,diff,pi,arctan2,sin,cos,append,max,min
from pandas import concat, read_csv
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
from TTutils.Utils import WindowFilter
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from TTutils.logo import *
from collections import OrderedDict
import matplotlib.colors as colors

sns.set_style("whitegrid", {
	                         'grid.color'    : ".5",
	                         'axes.edgecolor': "0",
	                         'legend.frameon': True,
							 'axes.linewidth': 1.5,
							 'axes.grid': True,
							 'grid.color': "#D2D2D2",
							 'grid.linestyle': u'-'
	                       }
	         )

sns.set_context("paper", rc={
							"font.size":8,
							"axes.titlesize":8,
							"axes.labelsize":8
							}
				)

class NearField():


	def __init__(self,
				ScenarioName,
				DischageDepth=None,
				RefDepth=1000,
				Concentration=False,
				CormixOutDatapath='c:/Program Files (x86)/CORMIX 8.0/System/'):

		if Concentration:
			self._Property='C'
		else:
			self._Property='S'

		self._Fnames =glob(ScenarioName+'*.txt')+glob(ScenarioName+'*.csv')

		self._RefDepth=RefDepth

		self._DischageDepth=DischageDepth

	@staticmethod
	def AdjustB(df):

		if 'B' in df.columns:
			df['BH']=df['B']
			df['BV']=df['B']
			df.drop('B',axis=1,inplace=True)

		df['BH']=df['BH']/2.

		return df

	@staticmethod
	def RemoveUcBo(df):

		if 'UC' in df.columns:
			df.drop('UC',axis=1,inplace=True)
		if 'Uc' in df.columns:
			df.drop('Uc',axis=1,inplace=True)
		if 'Bo' in df.columns:
			df.drop('Bo',axis=1,inplace=True)
		return df

	@staticmethod
	def RemoveZs(df):

		if 'ZU' in df.columns:
			df.drop(['ZU','ZL'],axis=1,inplace=True)
		return df

	@staticmethod
	def InterpolateNearField(df):

		df.index=ceil(df.index*100.)/10.
		df=df.groupby(df.index).first()
		N=int(ceil(df.index[-1]))+1
		df = df.reindex(arange(0,N,0.1)).interpolate(method='slinear')
		df.dropna(inplace=True)
		return df

	@staticmethod
	def CorrectsZ(df,RefDepth):

		df['Z']-=RefDepth
		df['Z']=abs(df['Z'])

		return df

	@staticmethod
	def AdjustTime(df):

		if df.index[0]==df.index[-1]:
			df = df.iloc[1:,:]

		return df

	@staticmethod
	def CalculatesB(df,RefDepth,ZeroCut=True):
		# calculates the B values
		df['BVinf']=df['Z']+df['BV']
		df['BVsup']=df['Z']-df['BV']

		# check what goes above surface ou bellow the floor
		out = df['BVinf'].values-RefDepth
		sup = df['BVsup'].values*1.
		if ZeroCut:
			out[out<0]=0
			sup[sup>0]=0

		# Put B values to other B if necessary
		df['BVsup']-=out
		df['BVinf']-=out
		df['BVsup']+=abs(sup)

		# Calcula o angulo da pluma
		theta = (pi)/2.-arctan2(diff(df['X']),diff(df['Y']))

		df['BX'] = append(0,sin(theta))*df['BH']
		df['BY']  = append(0,cos(theta))*df['BH']

		return df

	@staticmethod
	def ToFloat(df):

		for col in df.columns:
			df.loc[:,col]=df.loc[:,col].astype(float)
		df.index=df.index.astype(float)
		return df

	@staticmethod
	def ThetaCalc(df):

		Xdiff = df['X'].diff().values
		Ydiff = df['Y'].diff().values
		df['theta'] = np.arctan2(Xdiff,Ydiff)
		df['theta'][0]=0

		return df

	def ReadNearFieldFiles(self,**kwargs):

		nf=[]
		if 'sep' not in kwargs.keys():
			kwargs['sep']=';'

		for fname in self._Fnames:
			nf.append(read_csv(fname,index_col='TT',**kwargs)
				.pipe(self.RemoveZs)
				.pipe(self.AdjustB)
				.pipe(self.AdjustTime)
				.pipe(self.RemoveUcBo)
				.pipe(self.ToFloat)
			)

		df = concat(nf)

		df = self.ThetaCalc(df)

		df = df.pipe(self.InterpolateNearField)


		if not self._RefDepth:
			self._RefDepth = df.loc[:,'Z'].max()

		df.loc[:,'Z']=abs(df.loc[:,'Z'])

		df['Dist']=((df.loc[:,'X']**2)+(df.loc[:,'Y']**2))**0.5

		df.drop_duplicates('Dist',inplace=True)

		df = df.pipe(self.CorrectsZ,RefDepth=self._RefDepth)

		if not self._DischageDepth:
			self._DischageDepth=df['Z'].iloc[0]

		df = df.pipe(self.CalculatesB,RefDepth=self._RefDepth)

		df.iloc[1:,:] = df.iloc[1:,:].apply(WindowFilter)

		self.df=df

		return self

	def PlotProfile(self,ax1=None,ax2=None,ax3=None,**kwargs):

		axis = [ax1,ax2,ax3]
		if any(not ax for ax in axis):
			fig1, ax1 = plt.subplots(nrows=1, ncols=1)
			fig2, ax2 = plt.subplots(nrows=1, ncols=1)
			fig3, ax3 = plt.subplots(nrows=1, ncols=1)
			figs=[fig1,fig2,fig3]

		keys = kwargs.keys()

		if 'lw' not in keys:
			kwargs['lw']=2.

		if 'linestyle' in keys:
			linestyle=kwargs['linestyle']
			kwargs['linestyle']='solid'
		#plots
		ax1.plot(self.df['Dist'],self.df[self._Property],**kwargs)
		ax2.plot(self.df['Dist'],self.df['Z'],**kwargs)
		ax3.plot(self.df['X'],self.df['Y'],**kwargs)

		if 'linestyle' in keys:
			kwargs['linestyle']=linestyle
		else:
			kwargs['linestyle']='dashed'

		if 'label' in keys:
			del kwargs['label']

		kwargs['lw']=kwargs['lw']/2.

		ax2.plot(self.df['Dist'],self.df['BVsup'].values,**kwargs)
		ax2.plot(self.df['Dist'],self.df['BVinf'].values,**kwargs)

		Yel=np.sin(self.df['theta'].values)

		ax3.plot(self.df['X'].values,
				self.df['Y'].values+self.df['BH'].values*Yel,
				**kwargs)

		ax3.plot(self.df['X'].values,
				self.df['Y'].values-self.df['BH'].values*Yel,
				**kwargs)

		axis=[ax1,ax2,ax3]

		if 'figs' in dir():
			return [figs,axis]
		else:
			return axis
	@staticmethod
	def Plot3D(df,
				fname=None,
				Property='S',
				angle1=30,
				angle2=150,
				LongInterp=1000,
				LateralInterp=25,
				Dissolution=True,
				CMpallete=plt.cm.jet_r,
				autoscale=True,
				floor=None,
				**kwargs
				):

		if 'xlabel' not in kwargs.keys():
			xlabel = u'direção paralela \n a corrente (m)'
		else:
			xlabel=kwargs['xlabel']
		if 'ylabel' not in kwargs.keys():
			ylabel = u'direção perpendicular \n a corrente (m)'
		else:
			ylabel=kwargs['ylabel']
		if 'zlabel' not in kwargs.keys():
			zlabel = u'profundidade (m)'
		else:
			zlabel=kwargs['zlabel']
		if 'BarLabel' not in kwargs.keys():
			BarLabel = u'diluição (vezes)'
		else:
			BarLabel=kwargs['BarLabel']

		N=df.shape[0]

		idx = list(np.linspace(0,N-1,LongInterp))
		idx = list(map(int,idx))
		PlotDF=df.iloc[idx,:]
		PlotDF['X'][0]=0
		PlotDF['Y'][0]=0
		PlotDF['BV'][0]=0
		PlotDF['BX'][0]=0
		PlotDF['BY'][0]=0

		N=PlotDF['X'].shape[0]

		filter = signal.get_window(('gaussian',LateralInterp),LateralInterp)
		GridX,_ = np.meshgrid(PlotDF['X'],filter)
		GridY,_ = np.meshgrid(PlotDF['Y'],filter)
		GridZ,_ = np.meshgrid(PlotDF['Z'],filter)
		thetagrid = np.linspace(0,np.pi*2,len(filter))
		GridDissolution,_ = np.meshgrid(PlotDF[Property],filter)
		#GridDissolution=GridDissolution.T
		# TampaX=np.meshgrid(np.ones(N)*PlotDF['X'].iloc[-1],filter)[0]
		# TampaY=np.meshgrid(np.ones(N)*PlotDF['Y'].iloc[-1],filter)[0]
		# TampaZ=np.meshgrid(np.ones(N)*PlotDF['Z'].iloc[-1],filter)[0]



		for i in range(N):
		    GridX[:,i]+= np.cos(thetagrid)*(PlotDF.loc[:,'BX'].iloc[i])
		    GridY[:,i]+= np.cos(thetagrid)*(PlotDF.loc[:,'BY'].iloc[i])
		    if floor:
		    	if PlotDF['Z'].iloc[i]>floor:
		    		GridZ[:,i]=floor
		    	else:
		    		GridZ[:,i]+=np.sin(thetagrid)*(PlotDF.loc[:,'BV'].iloc[i])
		    else:
		    	GridZ[:,i]+=np.sin(thetagrid)*(PlotDF.loc[:,'BV'].iloc[i])
		    # TampaX[:,i]+= np.cos(thetagrid)*(PlotDF.loc[:,'BX'].iloc[-1])*fac
		    # TampaY[:,i]+= np.cos(thetagrid)*(PlotDF.loc[:,'BY'].iloc[-1])*fac
		    # TampaZ[:,i]+= np.sin(thetagrid)*(PlotDF.loc[:,'BV'].iloc[-1])*fac


		GridZ[GridZ<0]=0
		n=GridDissolution
		if Dissolution:
			n=n/GridDissolution.max()
		vmax=GridDissolution.max()

		cm = CMpallete(n)

		fig = plt.figure()
		ax = fig.add_subplot(111,
							projection='3d',
							)

		surf = ax.plot_surface(GridX,
		                       GridY,
		                       GridZ,
		                       facecolors=cm,
		                       linewidth=0.5,
		                       rstride=1,
		                       cstride=1,
		                       antialiased=False,
		                       shade=False,
		                       vmin=0,
		                       vmax=vmax
		                        )
		ax.invert_zaxis()
		m = plt.cm.ScalarMappable(cmap=CMpallete)
		m.set_clim(0,vmax)
		m.set_array(GridDissolution*100)
		cbar  = plt.colorbar(m,pad=0.1)
		cbar.set_clim(0,vmax)
		cbar.set_label(BarLabel,fontsize=8)
		if 'CbarTicks' in kwargs.keys():
			cbar.ax.set_yticklabels(kwargs['CbarTicks'])
		else:
			cbar.ax.set_yticklabels(list(map(str,list(np.arange(0,1,0.1)*int(vmax)))))

		ax.set_xlabel(xlabel,fontsize=8)
		ax.set_ylabel(ylabel,fontsize=8)
		ax.set_zlabel(zlabel,fontsize=8)
		ax.view_init(angle1, angle2)

		if 'xmax' not in kwargs.keys():
			xmax = max(GridX)
		else:
			xmax=kwargs['xmax']
		if 'ymax' not in kwargs.keys():
			ymax = max(GridY)
		else:
			ymax=kwargs['ymax']
		if 'zmax' not in kwargs.keys():
			zmax = max(GridZ)
		else:
			zmax=kwargs['zmax']
		if 'xmin' not in kwargs.keys():
			xmin = min(GridX)
		else:
			xmin=kwargs['xmin']
		if 'ymin' not in kwargs.keys():
			ymin = min(GridY)
		else:
			ymin=kwargs['ymin']
		if 'zmin' not in kwargs.keys():
			zmin = min(GridZ)
			if zmin<3:
				zmin=-5
		else:
			zmin=kwargs['zmin']

		if autoscale:

			xdist = xmax-xmin
			ydist = ymax-ymin
			zdist = zmax-zmin

			if xdist >= ydist > zdist:
				xydif = (xdist-ydist)/2
				ymin-=xydif
				ymax+=xydif

				xzdif = (xdist-zdist)/2
				zmin-=xzdif
				zmax+=xzdif
				if zmin<-5:
					zbase=zmin+5
					zmin-=zbase
					zmax+=zbase

			elif xdist <= ydist < zdist:
				xydif = (zdist-xdist)/2
				xmin-=xydif
				xmax+=xydif
				xydif = (zdist-ydist)/2
				ymin-=xydif
				ymax+=xydif

			elif ydist >= xdist > zdist:
				xydif = (ydist-xdist)/2
				xmin-=xydif
				xmax+=xydif

				yzdif = (ydist-zdist)/2
				zmin-=yzdif
				zmax+=yzdif
				if zmin<-5:
					zbase=zmin+5
					zmin-=zbase
					zmax+=zbase

		ax.auto_scale_xyz([xmin, xmax], [ymin, ymax], [zmin, zmax])


		a = plt.axes([.02, .02, .15, .15], facecolor='None')
		im = plt.imshow(np.array(Image.open(GetLogo())))
		plt.axis('off')
		plt.setp(a, xticks=[], yticks=[])
		if not fname:
			plt.show()
		else:
			fig.savefig(fname+'_3dplot.png',dpi=300)
			plt.close(fig)
		return GridDissolution
