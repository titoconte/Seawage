from geopandas import GeoDataFrame, read_file
from shapely.geometry import Point,LineString
from pandas import concat
from numpy import arange,where,append

def CreateShapefile(x,y,prof,name,epsg=4674,outputname='PontoDeModelagem.shp'):
	'''
	Create a shapefile with one or more points follow some procedures patterns

	input:
		x, y: float os list
			Points Position
		prof: float, int or list
			Depth list (one float or int value for each point) or
			float/int with depth
		name: str of list
			List of point names or string if just one name
		epsg: int or str, default 4674
			Epsg code the default projection is SIRGAS 200 geographic
		outputaname str, default 'PontoDeModelagem.shp'
			Name of shapefile output
	output:
		save a shapefile in a directory
	'''

	if isinstance(x,list):
		geometry=[Point(xx,yy) for xx,yy, in zip(x,y)]
	else:
		geometry=[Point(x,y)]
	gdf = GeoDataFrame(
		[{'Name':name,'prof':prof}],
		geometry=geometry,
		crs={'init':'epsg:{}'.format(epsg)}
	)

	gdf.to_file(outputname)

def DefineEPSGCode(lon):
	'''
	Gets UTM zone form EPSG code

	lon: int or float
		Longitude value
	output:
		Utm Zone number
	'''

	UTMLimits=arange(-180,180,6)
	UTMLimits=append(UTMLimits,lon)
	UTMLimits.sort()
	UTMZone = where(UTMLimits==float(lon))[0]

	return int(UTMZone)

def CreateRatiosFromPoint(fname,
				ratios=[100,200,300,400,500,600],
				outputname='Raios.shp',NewEPSG=None):
	'''
	Creates a line shapefile with ratios throgth buffer point shapefile

	inputs:
		fname: str
			shapefile name (must point)
		ratios: list, int or float , default [100,200,300,400,500,600]
			value of ratio distance in meters or a list of then
		outputaname: str, default 'Raios.shp'
			Name of shapefile output
		NewEPSG: None, int, str, default None
			Epsg conversion. If the epsg value is not passed or None
			the shapefile will not converted to other datum
	output:
		create a ratio shapefile
	'''
	# reads filename
	gdf = read_file(fname)

	crs = gdf.crs
	geometry = []
	final = []
	lon = gdf.geometry.bounds['minx']
	UTMZone = DefineEPSGCode(lon)
	epsg = {'init':'epsg:327{}'.format(UTMZone)}
	gdf = gdf.to_crs(epsg)
	for ratio in  ratios:
		geometry.append(
			LineString(
				gdf.geometry.buffer(ratio).values[0].exterior)
				)
		gdf['ratio']=ratio
		final.append(gdf.drop('geometry',axis=1))
		gdf.drop('ratio',axis=1)
	df = concat(final)
	df['geometry']=geometry
	gdf.reset_index(drop=True,inplace=True,)
	gdf = GeoDataFrame(df,crs=epsg)

	if isinstance(NewEPSG,(str,int)):

		gdf=gdf.to_crs({'init':'epsg:{}'.format(NewEPSG)})

	gdf.to_file(outputname)
