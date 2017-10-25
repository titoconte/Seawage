
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import geopandas as gpd
from glob import glob
from shapely.geometry import Polygon,Point,LineString
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import from_levels_and_colors
import datetime

sns.set_style("whitegrid", {'grid.color'    : ".5",
                                'axes.edgecolor': ".5",
                                }
                  )
sns.set_context('notebook',font_scale=1.5)

# output file name without extention
paths_sazo = ['inv','ver','out','pri',]
paths_kind = ['SLOP']
conc_inis = [1000000]
facfac=[80000]#,20000]
gambetinhas=[1]
#memb_ver 32.2 ; memb inv e pri: 28.8 ; memv out: 32.8


# Para rodar membrana com descartes alcalino e �cido no determin�stico ser� necess�rio identificar membflag com MEMB.
#MEMBsteps = [6,20]
MEMBsteps = [6,21]
membflag = 'MEMB'
MEMBCUT = [1,2,3,4,34,36,37,38,39,40,41]
# limites graficos perfil vertical
distlim = 10000
proflim = 150
acido_alcalino = False


# point coordinates do FPSO
rationame = '../Raios*.shp'
output_path = '../shapes/'

profs = [3,5,10]


class FarField():


    def __init__(scenario,
                 conc_ini=1000000,
                 DilutionLimit=10000,
                 GabrielFactor=1,
                 crs=4326):

        self.DilutionLimit=DilutionLimit

        self._GabrielFactor=GabrielFactor

        self._conc_ini=conc_ini

        self.crs = str(crs)

        self.Scenario=scenario

    @staticmethod
    def MatlabTime2PythonTime(MatlabTime):

        day = datetime.datetime.fromordinal(int(MatlabTime))
        DayFraction = datetime.timedelta(days=MatlabTime%1)
        DayEnd = day + DayFraction
        DayEnd = datetime.datetime(DayEnd.year-1,
                                   DayEnd.month,
                                   DayEnd.day,
                                   DayEnd.hour,
                                   DayEnd.minute,
                                   DayEnd.second
                                   )
        return DayEnd

    def GetsDischargeCell(self,DischargeFile=None):

        if not DischargeFile:
            DischargeFile='../discharge/{}.src'.format(self.Scenario)

        src=np.genfromtxt(self.Scenario)
    	m,n=src[2:4]

        self.Src = {'m':m,'n':n}

        return self

    def GetsCoordinates(self,GridFile):

        if not GridFile:
            '../domain/{}_grid.mat'.format(self.Scenario)

        mat = h5py.File(GridFile)
        lons = np.ma.masked_invalid(mat['x'].value)
    	lats = np.ma.masked_invalid(mat['y'].value)
    	clons = lons[:-1,:-1]+np.diff(lons[:,:-1],axis=0)
    	clats = lats[:-1,:-1]+np.diff(lats[:-1:,],axis=1)
        x=lon[m-1,n-1]
		y=lat[m-1,n-1]
		dx =x-plon
		dy =y-plat
		self.lon = lon-dx
		self.lat = lat-dy
		self.clon = clon-dx
		self.clat = clat-dy

        return self

    def CalculatesDistanceMatrix(self):

        distance = ((self.plon-self.clon)**2+(self.plat-self.clat)**2)**.5

        return self



def ratios_points(shp,rationame,outputname,kind='dilui',names='A B C D E F'.split(' ')):

	ratios = gpd.read_file(rationame)
	ratios[kind] = list(np.zeros(6))
	for i in range(ratios.shape[0]):
		cuted = shp[shp.intersects(ratios['geometry'][i])]
		cuted = cuted.sort_values(by=kind)
		if len(cuted)>0:
			try :
				x,y = cuted['geometry'].iloc[0].intersection(ratios['geometry'][i]).xy
			except:
				x,y = cuted['geometry'].iloc[0].intersection(ratios['geometry'][i])[0].xy
			x = x[int(np.ceil(len(x)/2))]
			y = y[int(np.ceil(len(y)/2))]
			ratios[kind].iloc[i] = np.round(cuted.sort_values(kind)[kind].iloc[0],0)
			ratios['geometry'].iloc[i] = Point(x,y)
	ratios = ratios.sort_values('ratio')
	ttname = ['{} - {}'.format(i,int(j)) for i,j in zip(names,ratios[kind].values)]
	ratios['NAME'] = ttname
	ratios['LAYER'] = names
	ratios = ratios[ratios[kind]>0]
	ratios=ratios.loc[ratios[kind]<999900]
	if len(ratios)>0:
		# save shape
		print(ratios[['NAME','ratio']])
		ratios.to_file(outputname)
		if '_det_' in outputname:
			sufix='_det.csv'
			ind=-4
		else:
			sufix='_prob.csv'
			ind=-3
		ratios.T.iloc[[-3,-1],:].to_csv('_'.join(outputname.split('_')[:ind])+sufix,header=False,mode='a')
		# make the projection file
		make_proj(outputname)
	else:
		print('nao toca os circulos')

def perfil_vertical(distance,d,profs,outputname,conc_cut,clon,clat):


	colors = ['#D93027','#FCC78B','#4575B5','#74B4E8']
	levels = [0,1000,2000,5000,10000]
	cmap, norm = from_levels_and_colors(levels, colors)

	# get the start values
	mi,ni = np.where(np.min(distance)==distance)

	diluidist = np.min(d,axis=0)
	distance = np.ma.array(distance,mask=diluidist.mask)
	mend,nend = np.where(np.max(distance)==distance)

	dm = mend-mi
	dn = nend-ni
	values = []

	if abs(dn)<abs(dm):
		if dm>0:
			m= np.arange(mi,mend)
		else:
			m= np.flipud(np.arange(mend,mi))

		if dn>0:
			n= np.round(np.linspace(ni,nend,abs(dm)))
		else:
			n = np.flipud(np.round(np.linspace(nend,ni,abs(dm))))

		dist = np.linspace(np.min(distance),np.max(distance),abs(dm))

	elif abs(dn)>=abs(dm):
		if dm>0:
			m= np.round(np.linspace(mi,mend,abs(dn)))
		else:
			m= np.flipud(np.round(np.linspace(mend,mi,abs(dn))))

		if dn>0:
			n= np.arange(ni,nend)
		else:
			n = np.flipud(np.arange(nend,ni))

		dist = np.linspace(np.min(distance),np.max(distance),abs(dn))


	for mm,nn in zip(m,n):values.append(d[:,mm,nn])

	shp = gpd.GeoDataFrame([],columns=['NAME','geometry'],index=[0])
	shp['geometry'][0] = LineString([(clon[mi,ni],clat[mi,ni]),(clon[mend,nend],clat[mend,nend])])
	shp['NAME'][0]  = 'transect'

	shp.to_file(outputname.replace('det','det_transect'))
	values = np.ma.array(values)

	P,D = np.meshgrid(profs,dist)

	D =D*1852*60
	ax = plt.subplot(111)
	plt.pcolor(D,P,values,cmap=cmap,norm=norm)
	cb = plt.colorbar()
	cb.ax.set_ylabel(u'Dilui��o')
	ax.set_xlabel(u'Dist�ncia do ponto de descarte (m)')
	ax.set_ylabel(u'Profundidade (m)')
	ax.set_ylim(proflim,0)
	ax.set_xlim(0,distlim)
	plt.savefig(outputname.replace('.shp','_perfil.png'),dpi=300,bbox_inches='tight')
	#plt.show()
	plt.close()

def det_shape(step,matname,maxname,conc_cut,conc_ini,clon,clat,profs,distance,rationame,f,names='A B C D E F'.split(' ')):

	# carregar os mats
	mat = h5py.File(matname)

	# get the information from mat file and mask the invalid values
	concentration = np.ma.masked_invalid(mat['d'].value)*gambi

	# get the step time
	nt = concentration.shape[0]

	#print(nt)
	# apendar resultado da concentra��o m�dia no tempo para cada ponto
	d = conc_ini/np.ma.masked_less(concentration,conc_ini/10000.)

	# cutd
	d=d[:,1:-1,1:-1,:]

	# inserir na lista das simula��es

	simus1 = np.min(np.min(d,axis=3),axis=0)

	#simus2 = np.min(np.mean(d2,axis=0),axis=2)

	validos = np.where(abs(simus1.mask-1)==1)

	#validos2 =np.where(abs(simus2.mask-1)==1)

	# make the geographic dataframe object
	shp = gpd.GeoDataFrame([],columns=['dilui','geometry'],index=range(len(validos[0])))
	# write information in geographic dataframe object
	for k,i,j in zip(range(len(validos[0])),validos[0],validos[1]):
			print(i,j,k)
			shp['geometry'][k] 	= Polygon([(lon[i,j],lat[i,j]),(lon[i+1,j],lat[i+1,j]),(lon[i+1,j+1],lat[i+1,j+1]),(lon[i,j+1],lat[i,j+1]),(lon[i,j],lat[i,j])])
			shp['dilui'][k] 	= simus1[i,j]

	shp = shp.dropna()

	# format shape atributes
	shp['dilui'] 	= shp['dilui'].astype(float)

	# save shape
	shp.to_file(maxname.replace('det','det_varrida'))
	# make the projection file
	make_proj(maxname.replace('det','det_varrida').replace('.shp',''))
	#f.write(u'Simulation with the max {} value:'.format(step))

	invalidos = np.where(simus1.mask==1)

	# verifica se o step � ce concentra��o maxima ou dist maxima ou nenhum (ai � o ultimo step)
	if step=='conc':
		# calcular step com a minima dilui��o
		step = np.argmin(np.min(d,axis=(1,2,0)))
	elif step=='dist':
		# indice da maior distancia dos valres da area varrida
		indmax = np.argmax(distance[validos[0],validos[1]])

		# calcular step com a maxima distancia
		step = np.argmin(np.min(d,axis=0)[validos[0][indmax],validos[1][indmax],:])
	else:
		step=-1


	# inserir na lista das simula��es
	simus = np.squeeze(np.min(d,axis=0)[:,:,step])
	simutime = matlabtime2pythontime(np.squeeze(mat['t'].value)[step])
	f.write(u'\t{}\t{}\n'.format(matname,simutime.strftime('%d/%m/%Y %H:%M')))

	validos = np.where(abs(simus.mask-1)==1)
	# make the geographic dataframe object
	shp = gpd.GeoDataFrame([],columns=['dilui','geometry'],index=range(len(validos[0])))
	# write information in geographic dataframe object
	for k,i,j in zip(range(len(validos[0])),validos[0],validos[1]):
		shp['geometry'][k] 	= Polygon([(lon[i,j],lat[i,j]),(lon[i+1,j],lat[i+1,j]),(lon[i+1,j+1],lat[i+1,j+1]),(lon[i,j+1],lat[i,j+1]),(lon[i,j],lat[i,j])])
		shp['dilui'][k] 	= simus[i,j]

	# format shape atributes
	shp['dilui'] 	= shp['dilui'].astype(float)
	# save shape
	shp.to_file(maxname)
	# make the projection file
	make_proj(maxname[:-4])

	perfil_vertical(distance,np.squeeze(d[:,:,:,step]),profs,maxname,conc_cut,clon,clat)
	print('shape_det escrito conc')
	ratios_points(shp,rationame,maxname.replace('det','det_pts'),names=names)

#sys.exit()
if __name__=='__main__':

	rationame = glob(rationame)[0]

	corr=gpd.read_file('../FPSO.shp')
	[plon,plat]=corr.geometry.loc[0].xy

	f = open('diagnostico.txt','a+')
	for path1,conc_ini,ddst,gambi in zip(paths_kind,conc_inis,facfac,gambetinhas):
		# src=np.genfromtxt('../discharge/{}.src'.format(path1))
		# m,n=src[2:4]
		# gets the grid index
		lon,lat,clon,clat = read_grid_waq('../domain/{}_grid.mat'.format(path1))
		# subtracts two because we use open boundaries, and the numerical grid is centered
		x=lon[m-1,n-1]
		y=lat[m-1,n-1]
		dx =x-plon
		dy =y-plat
		lon = lon-dx
		lat = lat-dy
		clon = clon-dx
		clat = clat-dy
		nx,ny=clon.shape
		# calculates distance matrix
		distance = ((plon-clon)**2+(plat-clat)**2)**.5
		for path2 in paths_sazo:

			conc_cut = conc_ini/ddst  # float((raw_input(u'Concentra��o de corte em g/m^3 [100]') or '100'))
			# math path name
			matpath='{}/{}/'.format(path1,path2)
			print(matpath)
			# outname
			outputname = '{}{}prob.shp'.format(output_path,matpath.replace('/','_'))

			# get the matfiles in the directory
			arqs = glob(matpath+'*.mat')
			# numero de simulacoes
			nsim = len(arqs)

			# criar eixo com as simula��es
			simus = np.ma.masked_array(np.zeros((len(arqs),nx,ny)))

			maxidist = []
			maxiconclist = []
			nz=len(profs)
			maxprof = np.zeros(nz)

			mtimes=[]

			# loop sobre os arquivos
			for i,arq in enumerate(arqs):

				# apresentar a simula��o que esta sendo lida
#				try:
				# carregar os mats
				mat = h5py.File(arq)

				# get the informatino from mat file and mask the invalid values
				d = np.ma.masked_invalid(mat['d'].value)*gambi

				# cutd
				d=d[:,1:-1,1:-1,:]

				mtimes.append(matlabtime2pythontime(np.min(mat['t'].value)))

				# get the step time
				nt = d.shape[0]

				# gets the fill.value
				fv = d.fill_value

				# obter a presen�a de efluente em casa camada
				mprof = np.sum(np.sum(np.sum(abs(d.mask-1),axis=1),axis=1),axis=1)*np.ones(nz)>0
				# inserir no maximo
				maxprof += mprof

				# verificar a distancia e inserir na lista
				diluidist = np.max(np.max(d,axis=3),axis=0)
				diluidist = np.ma.masked_less(diluidist,conc_ini/10000.)
				SimuDist = np.max(np.ma.array(distance,mask=diluidist.mask))
				if np.ma.is_masked(SimuDist):
					SimuDist=0
				maxidist.append(SimuDist)
				maxiconclist.append(np.max(np.mean(d,axis=3)))


				if path1==membflag:
					simus[i,:,:] = np.max(np.mean(d[MEMBCUT,:,:,:],axis=3),axis=0)
				else:
					simus[i,:,:] = np.max(np.mean(d,axis=3),axis=0)
#				except:
#					print(U'\n\n\n\nATEN��O CEN�RIO COM ERRO\n\n\n\n\n\n')
#					print(arq)

			simus = np.ma.masked_less(simus,conc_cut)
			# fazer os diagnosticos
			# gets the scenario with maximum concentration
			maxisim  = arqs[np.argmax(np.asarray(maxiconclist))]
			conctime  = mtimes[np.argmax(np.asarray(maxiconclist))]
			# gets the scenario with maximum distance
			disttime  = mtimes[np.argmax(np.asarray(maxidist))]
			maxdist  = arqs[np.argmax(np.asarray(maxidist))]
			print 'distancia: {}'.format(maxdist)
			print 'concentracao: {}'.format(maxisim)
			# checks the limits depth in all simulation
			camdmax = np.argmax((maxprof*np.arange(1,nz+1))/maxprof)+1

			# integrates the results in all vertical layers
			tot_max = conc_ini/np.max(simus,axis=0)
			tot_med = conc_ini/np.mean(simus,axis=0)

			# calculates the probability
			prob  = 100*np.mean(abs(np.asarray(simus.mask-1)),axis=0)

			validos = np.where(tot_max.mask==False)

			# make the geographic dataframe object
			shp = gpd.GeoDataFrame([],columns=['prob_med','prob_min','dilui_min','dilui_med','geometry'],index=range(len(validos[0])))

			# write information in geographic dataframe object
			for k,i,j in zip(range(len(validos[0])),validos[0],validos[1]):
				shp['geometry'][k] 	= Polygon([(lon[i,j],lat[i,j]),(lon[i+1,j],lat[i+1,j]),(lon[i+1,j+1],lat[i+1,j+1]),(lon[i,j+1],lat[i,j+1]),(lon[i,j],lat[i,j])])
				shp['dilui_min'][k] 	= tot_max[i,j]
				shp['prob_min'][k] 		= prob[i,j]
				if tot_max[i,j]==tot_med[i,j]:
					shp['dilui_med'][k] 	= 999999
					shp['prob_med'][k] 		= -99999
				else:
					shp['dilui_med'][k] 	= tot_med[i,j]
					shp['prob_med'][k] 		= prob[i,j]

			# format shape atributes
			shp['prob_med']		= shp['prob_med'].astype(float)
			shp['prob_min']		= shp['prob_min'].astype(float)
			shp['dilui_med'] 	= shp['dilui_med'].astype(float)
			shp['dilui_min'] 	= shp['dilui_min'].astype(float)
			# save shape
			print('shape de probabilidade escrito')
			shp.to_file(outputname)
			# make the projection file
			make_proj(outputname.replace('.shp',''))

			ratios_points(shp,glob(rationame)[0],outputname.replace('prob','pts_diluimed'),kind='dilui_med')
			print('raio escrito')
			ratios_points(shp,glob(rationame)[0],outputname.replace('prob','pts_diluimin'),kind='dilui_min')
			print('raio escrito')
			if path1!=membflag:
				det_shape('conc',maxisim,outputname.replace('prob','det_diluimin'),conc_cut,conc_ini,clon,clat,profs,distance,rationame,f)
				det_shape('dist',maxdist,outputname.replace('prob','det_distmax'),conc_cut,conc_ini,clon,clat,profs,distance,rationame,f,names='K L M N O P'.split(' '))
			elif acido_alcalino:
				for tipo,step in zip(['acido','alcalino'],MEMBsteps):
					det_shape('conc',maxisim,'diluimin_'+tipo,matpath,conc_cut,conc_ini,clon,clat,profs,distance,rationame,f,step=step)
					det_shape('dist',maxdist,'distmax_'+tipo,matpath,conc_cut,conc_ini,clon,clat,profs,distance,rationame,f,names='K L M N O P'.split(' '),step=step)
			else:
				det_shape('conc',maxisim,outputname.replace('prob','det_diluimin'),conc_cut,conc_ini,clon,clat,profs,distance,rationame,f)
				det_shape('dist',maxdist,outputname.replace('prob','det_distmax'),conc_cut,conc_ini,clon,clat,profs,distance,rationame,f,names='K L M N O P'.split(' '))
	f.close()
	aa=input('pres enter to finish')
