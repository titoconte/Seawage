

import os
import shutil
from glob import glob
from datetime import datetime
import time


class Products():

    def __init__(self,path):

        self.path = path

        self.Company='PETROBRAS'
        self.Basin='BSant'

        self.EfluentType = {
            'Efl_AgProd':'EFLUENTE DE ÁGUA PRODUZIDA - VAZÃO CORRESPONDENTE À CAPACIDADE DE TRATAMENTO DA UNIDADE;',
            'Efl_AgProd80':'EFLUENTE DE ÁGUA PRODUZIDA - VAZÃO CORRESPONDENTE À 80% DA CAPACIDADE DE TRATAMENTO DA UNIDADE;',
            'Efl_AgProd2037':'EFLUENTE DE ÁGUA PRODUZIDA - VAZÃO CORRESPONDENTE À GERAÇÃO ESTIMADA PARA 2037;',
            'Efl_SLOP':'EFLUENTE DE DESCARTE VIA TANQUE SLOP;',
            'Efl_URS':'EFLUENTE DE DESCARTE DA URS;',
            }



    def SetProtocolNumber(self,ProtocolNumber):
        self.ProtocolNumber=ProtocolNumber
        return self

    def SetCompanyName(self,Company):
        self.Company=Company
        return self

    def SetPointName(self,PointName):
        self.PointName=PointName
        return self

    def BasinDetection(self,Basin):
        # self.Basin=Basin
        # return self
        pass

    @staticmethod
    def Png2A3pdf(TargetDirectory,removepng=True):
        shutil.copy('U:/png2pdf/pscp.exe',TargetDirectory)
        shutil.copy('U:/png2pdf/plink.exe',TargetDirectory)
        shutil.copy('U:/png2pdf/png2A3.bat',TargetDirectory)
        os.system(TargetDirectory+'/png2A3.bat')
        for exefiles in glob(TargetDirectory+'/*.exe'):
            os.remove(exefiles)
        for exefiles in glob(TargetDirectory+'/*.sh'):
            os.remove(exefiles)
        os.remove(TargetDirectory+'/png2A3.bat')
        if removepng:
            for fname in glob(TargetDirectory+'/**/*.png',recursive=True):
                os.remove(fname)


    def CreateReadme(self,path):
        today = datetime.today().strftime('%d/%m/%Y')
        f.write('Leia-me.txt TetraTech '+today)
        f.write('\r\n---------------------------------------------------------------------------')
        f.write('\r\n')
        f.write('A Pasta SHAPES é composta por subpastas:')
        f.write('\r\n')

        dirs = [os.path.normpath(i).split(os.sep)[-1] for i in glob(self.Shapes+'*/*/')]

        fnames = [os.path.split(ff)[-1] for ff in glob(self.Shapes+'**/*.shp',recursive=True)]
        fname = fnames[0]

        keys=['_'.join(ff.replace(self.prefix+'_','').split('_')[:2]) for ff in fnames]

        f.write('- PROBABILISTICOS:')
        f.write('\r\n')
        for d in dirs
            f.write('		- {}:\r\n'.format(d))
        f.write('- DETERMINISTICOS:')
        f.write('\r\n')
        for d in dirs
            f.write('		- {}:\r\n'.format(d))

        f.write('- e um arquivo leia_me.txt\r\n')
        f.write('Os arquivos nestas subpasta são entitulados:\r\n')
        f.write('Exemplos:\r\n')
        f.write('EX.1 - PROBABILISTICOS: "{}"\r\n'.format(fname))
        f.write('Sendo composto por:\r\n')
        f.write('#1. Empresa: {}\r\n'.format(self.Company))
        f.write('#2. Número de referência: {}\r\n'.format(self.ProtocolNumber))
        f.write('#3. Território de Abrangência: {}\r\n'.format(self.Basin))
        f.write('#4. Fonte do dado: {}\r\n'.format(self.Tt))
        f.write('#5. Data de aquisição: {}\r\n'.format(self.today))
        f.write('#6. Tema do mapeamento (exemplo): {}\r\n'.format(self.PointName))
        f.write('#7. Referencias adicionais (exemplos):\r\n')

        for key in keys:
            val = self.EfluentType[key]
            f.write('\t\t- {}\t\t\t\t\t: {}\r\n'.format(key,val))

        f.write('\t\t\t- PROB\t\t\t\t: CENARIO PROBABILISTICO;\r\n \
\t\t\t- DET\t\t\t\t: CENARIO DETERMINISTICOS;\r\n\r\n\t\t\t- VER\t\t\t\t\: SAZON\
ALIDADE SIMULADA DE VERÃO 	(janeiro a março);\r\n\t\t- OUT\t\t\t\t: SAZONALIDAD\
E SIMULADA DE OUTONO 	(abril a junho);\r\n\t\t- INV\t\t\t\t: SAZONALIDADE SIMU\
LADA DE INVERNO 	(julho a setembro);\r\n\t\t- PRI\t\t\t\t: SAZONALIDADE SIMUL\
ADA DE PRIMAVERA 	(outubro a dezembro);\r\n\r\n\t\t- DILUI\t\t\t\t: SHAPE DETE\
RMINISTICO DO CENÁRIO DE MENOR DILUIÇÃO DA PLUMA;\r\n\t\t- DIST\t\t\t\t: SHAPE D\
ETERMINISTICO DO CENÁRIO DE MAIOR DISTÂNCIA ATINGIDA PELA PLUMA;\r\n\t\t- Instan\
te\t\t\t: PLUMA DO EFLUENTE NO INSTANTE FINAL DO DERRAME;\r\n\t\t- Area\t\t\t\t: \
ÁREA DE DESLOCAMENTO DA PLUMA DO EFLUENTE AO LONGO DA SIMULAÇÃO DETERMINISTICA;\r\n\
\r\n---------------------------------------------------------------------------')
        f.write('As SUB_PASTAS (citadas acima) contêm:\r\n')
        f.write('#1 (.dbf)\r\n')
        f.write('#2 (.prj)\r\n')
        f.write('#3 (.shp)\r\n')
        f.write('#4 (.shx)\r\n')
        f.write('___________________________________________________________________________')
        f.write('\r\n')
        f.write('#1 PROBABILISTICOS\r\n')
        f.write('referência adicionais da tabela de atributos dos shapes:\r\n')
        f.write('prob_min 	: Probabilidade total na coluna da água em porcentagem (%) associada pluma de diluição mínima;\r\n')
        f.write('prob_med 	: Probabilidade total na coluna da água em porcentagem (%) associada a pluma de diluição média,\r\n')
        f.write('\t\tvalores iguais a -99999 correspondem a valores extrapolados no processamento;\r\n\r\n')
        f.write('dilui_min	: Diluição mínima do efluente na coluna da água em número de vezes (x) (ex.: 2489,0 vezes);\r\n')
        f.write('dilui_med	: Diluição média do efluente na coluna da água em numero de vezes (x) (ex.: 4389,0 vezes),\r\n')
        f.write('\t\tvalores iguais a -99999 correspondem a valores extrapolados no processamento;\r\n')
        f.write('\r\n')
        f.write('Topologia dos arquivos : Poligono')
        f.write('\r\n')
        f.write('\r\n')
        f.write('___________________________________________________________________________')
        f.write('\r\n')
        f.write('#2 DETERMINISTICOS\r\n')
        f.write('referência adicionais da tabela de atributos dos shapes:\r\n')
        f.write('No shape de INSTANTE FINAL (Instante):\r\n')
        f.write('dilui : Diluição mínima do efluente na coluna da água em número de vezes (x) (ex.: 2489,0 vezes) no instante final do derrame;\r\n\r\n')
        f.write('No shape de ÁREA DE DESLOCAMENTO (Area):\r\n')
        f.write('dilui : Valores de diluição criticos do efluente registrada para cada célula, ao longo de toda a simulação;\r\n')
        f.write('\r\n')
        f.write('Topologia dos arquivos : Poligono')
        f.write('\r\n')
        f.write('\r\n')
        f.write('___________________________________________________________________________')
        f.write('\r\n')
        f.write('Todos os shapes estão georeferenciados no Sistema de Coordenadas Geográficas DATUM Sirgas2000.')
        f.write('\r\n')
        f.write('\r\n---------------------------------------------------------------------------')


    @staticmethod
    def InsertsPrefix(prefix,filename,splitcharacter='_'):

        path,fname = os.path.split(filename)
        fname,extc = os.path.splitext(fname)
        nfilename = path+'/'+prefix+splitcharacter+fname.upper()+extc

        return nfilename

    def PrepareSeawageProducts(self):

        DirsNames = ['A_Documentos','B_Ilustracoes','C_Shapes']
        for prop in DirsNames:
            setattr(self,prop[2:],os.path.join(self.path,'Produtos/'+prop))

        IgnPerfis = shutil.ignore_patterns('PERFIS','*.db')
        IgnDocs = shutil.ignore_patterns('*uperados','*.docx','*.db')
        IgnCpg = shutil.ignore_patterns('*.cpg','*_pts_*','*_transect_*','*.db')
        shutil.copytree(
                        '../Figuras/Efluente/CampoAfastado/',
                        self.Ilustracoes+'/',
                        ignore=IgnPerfis
                        )
        shutil.copytree(
                        '../../Docs/',
                        self.Documentos+'/',
                        ignore=IgnDocs
                        )
        shutil.copytree(
                        '../Sig/Shapes/',
                        self.Shapes+'/',
                        ignore=IgnCpg
                        )

        shutil.copy('../Figuras/Localizacao.png',self.Ilustracoes)

        self.today = datetime.today().strftime('%Y_%m_%d')
        year = datetime.today().strftime('%Y')
        self.Tt = 'TetraTech'

        self.prefix='_'.join([
                        self.Company,
                        self.ProtocolNumber,
                        self.Basin,
                        self.Tt,self.today,
                        self.PointName,
                        'Efl'
                        ])

        filenames = glob(self.Ilustracoes+'/**/*.png',recursive=True)
        filenames.extend(glob(self.Shapes+'/**/*.shp',recursive=True))
        filenames.extend(glob(self.Shapes+'/**/*.shx',recursive=True))
        filenames.extend(glob(self.Shapes+'/**/*.prj',recursive=True))
        filenames.extend(glob(self.Shapes+'/**/*.dbf',recursive=True))

        for filename in filenames:
            nfilename = self.InsertsPrefix(self.prefix,filename)

            nfilename=nfilename.replace('VARRIDA','Area')
            nfilename=nfilename.replace('AGPROD','AgProd')
            nfilename=nfilename.replace('DET_D','DET_Instante_D')
            nfilename=nfilename.replace('FUT_',year+'_')
            nfilename=nfilename.replace('DILUIMIN','DILUI')
            nfilename=nfilename.replace('DISTMAX','DIST')
            nfilename=nfilename.replace('OVERBOARD','Overboard')
            nfilename=nfilename.replace('MEMB_SLOP','SLOP')
            nfilename=nfilename.replace('Localizacao'.upper(),'Localizacao')

            shutil.move(filename,nfilename)
