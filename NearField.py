
from glob import glob
from numpy import arange,ceil,abs,diff,pi,arctan2,sin,cos,append
from pandas import concat, read_csv
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation, TriAnalyzer, UniformTriRefiner
from TTutils.Utils import WindowFilter

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
				Concentration=False,):

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
	def CalculatesB(df,RefDepth):
		# calculates the B values
		df['BVinf']=df['Z']+df['BV']
		df['BVsup']=df['Z']-df['BV']

		# check what goes above surface ou bellow the floor
		out = df['BVinf'].values-RefDepth
		out[out<0]=0
		sup = df['BVsup'].values*1.
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
			)

		df = concat(nf)

		df = df.pipe(self.InterpolateNearField)

		if not self._RefDepth:
			self._RefDepth = df.loc[:,'Z'].max()

		df.loc[:,'Z']=abs(df.loc[:,'Z'])

		df['Dist']=((df.loc[:,'X']**2)+(df.loc[:,'Y']**2))**0.5

		df.drop_duplicates('Dist',inplace=True)

		df = df.pipe(self.CorrectsZ,RefDepth=self._RefDepth)
		if not self._DischageDepth:
			self._DischageDepth=df['Z'].loc[0]

		df = df.pipe(self.CalculatesB,RefDepth=self._RefDepth)

		df = df.apply(WindowFilter,WindowSize=100)

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

		ax3.plot(self.df['X'].values,
				self.df['Y'].values+self.df['BH'].values,
				**kwargs)

		ax3.plot(self.df['X'].values,
				self.df['Y'].values-self.df['BH'].values,
				**kwargs)

		axis=[ax1,ax2,ax3]

		if 'figs' in dir():
			return [figs,axis]
		else:
			return axis

	# def Plot3D(self,fname,angle,resolution):
	#
	#
	# 	filter = signal.get_window(('gaussian',25))
	# 	GridDissolution,_ = np.meshgrid(df_interp['C'],filter)
	# 	# espacial
	# 	# grid x e y
	# 	GridX,_ = np.meshgrid(df_interp['X'],filter)
	# 	GridY,_ = np.meshgrid(df_interp['Y'],filter)
	# 	GridZ,_= np.meshgrid(df_interp['Z'],filter)
	# 	thetagrid = np.linspace(0,2*np.pi,len(filter))
	# 	for i in range(resolution):
	# 		grid_x[:,i]= np.sin(thetagrid)*(df_interp['bx_pos'][i]-df_interp[xlabel][i])+df_interp[xlabel][i]
	# 		grid_y[:,i]= np.sin(thetagrid)*(df_interp['by_pos'][i]-df_interp[ylabel][i])+df_interp[ylabel][i]
	# 		grid_z[:,i]=np.cos(thetagrid)*(df_interp['bv_pos'][i]-df_interp[zlabel][i])+df_interp[zlabel][i]
	# 	center = int(resolution/2)
	# 	# retorna todas as grades
	# 	return grid_x,grid_y,grid_z,grid_dissolution
