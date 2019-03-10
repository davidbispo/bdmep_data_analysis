# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 14:28:26 2018

@author: David Bispo
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from scipy.interpolate import spline
#import geopandas as gpd
#from shapely.geometry import Polygon

"""O programa possui dois modulos:
    1) data_dictionary, info_array = getclimate(root) - Busca um dicionario com chaves. 

"""
def getclimate(root, vento = False):
    
    print('lendo dados das estacoes...')
    all_data = {}
    filelist_final = []
        
    filelist = os.listdir(root)
    for file in filelist:
        if file.endswith(".txt"):
            filelist_final.append(file)
    
    for j in filelist_final:
    
        filename = os.path.join(root, j)
    
        openfile = open(filename, 'r')
        lines = openfile.readlines()
        
        latitude = lines[4]
        finder_lat = latitude.find(":")
        latitude =  float(latitude[finder_lat+1:])
        
        longitude = lines[5]
        finder_lon = longitude.find(":")
        longitude =  float(longitude[finder_lon+1:])
        
        altitude = lines[6]
        finder_alt = altitude.find(":")
        altitude =  float(altitude[finder_alt+1:])
                
        operating = lines[7]
        inicio_operacao = lines[8].rstrip()
        finder_ini = inicio_operacao.find(":")
        inicio_operacao =  inicio_operacao[finder_ini+1:]
        
        station_name = lines[3]
        finder_nome = station_name.find(":")
        station_name =  station_name[finder_nome+1:].rstrip()
        
        station_data = {'nome' : station_name, 
                 'latitude':latitude, 'longitude': longitude, 'altitude': altitude, 'operating': operating, 'inicio_operacao' : inicio_operacao}
        
        data = lines[16:]
        
        parsed_data = []
        
        for i in data:
            parsed_line = i.split(";")
            parsed_data.append(parsed_line)
        
        del parsed_data[0][2]
        k=1
        
        for o in (parsed_data[1:]):
            timestamp_temp = "%s %s" % (o[1],o[2])
            timestamp_datetime = datetime.strptime(timestamp_temp, '%d/%m/%Y %H%M')
            o[1] = timestamp_datetime
            del o[2]
            parsed_data[k] = o
            k +=1
            
        header = parsed_data[0]
        df = pd.DataFrame(parsed_data[1:], columns=header)
        df.set_index('Data', inplace = True)
        
        station_data['data'] = df
        all_data[station_name] = station_data
           
    info_array = np.array([['nome'],['latitude'], ['longitude'], ['altitude'], ['inicio_operacao']]).T
        
    for c in all_data:
        
        keys = list(all_data[c]['data'].keys())
        all_data[c]['data'] = all_data[c]['data'].drop(['\n'], axis=1)
    
        estacao_index = keys.index("Estacao")
        del keys[estacao_index]
        
        if vento == True:
            if 'Temp Comp Media' not in keys:
                temp_maxima = all_data[c]['data']['TempMaxima']
                temp_maxima = temp_maxima.apply(pd.to_numeric, errors='coerce')
                temp_maxima_resample = temp_maxima.resample('D').mean()
                
                temp_minima = all_data[c]['data']['TempMinima']
                temp_minima = temp_minima.apply(pd.to_numeric, errors='coerce')
                temp_minima_resample = temp_minima.resample('D').mean()
                
                temp_media_comp = (temp_maxima_resample+temp_minima_resample)/2
                all_data[c]['data']['Temp Comp Media'] = temp_media_comp
               
        inicio_operacao = all_data[c]['data'].index[0]
        
        new_array = np.array([[(all_data[c]['nome'])],
                               [(all_data[c]['latitude'])], 
                               [(all_data[c]['longitude'])], 
                               [(all_data[c]['altitude'])], 
                               [(all_data[c]['inicio_operacao'])]]).T
        
        info_array = np.vstack((info_array, new_array))
        
    print('salvando consulta em arquivo csv...')
    print(info_array)
    np.savetxt("info_array.csv", info_array, delimiter=",",fmt='%s')
       
    print('Dicionario de estacoes terminado!')
    return all_data, info_array


def get_series_queried_singlevalueinfo(data_dictionary, info_array, key_to_output, startdate,enddate, plot_original = False):

        data_dictionary_keys = list(data_dictionary.keys())
        print('Comecando a printar dado para mapa...')
        for k in data_dictionary_keys:
            
            data_dictionary[k]['data'] = data_dictionary[k]['data'].loc[startdate:enddate]
            df_to_plot = data_dictionary[k]['data']
            df_to_plot_keys = list(df_to_plot.keys())
            
            estacao_index = df_to_plot_keys.index("Estacao")
            del df_to_plot_keys[estacao_index]
            
            for l in df_to_plot_keys:
            
                title = l + ' - ' + data_dictionary[k]['nome'] 
                
                df_to_plot_series = df_to_plot[l]
                df_to_plot_series = pd.to_numeric(df_to_plot_series)
                df_to_plot_series_resampled = df_to_plot_series.resample('D').mean() # Arrumando a serie
                
                #Saidas para mapas - append de coluna numpy array 
                if l == key_to_output:
############################################################################################
                    #escolha aqui o que quer que saia no mapa como um float :D
                    
                    media_serie_completa = df_to_plot_series_resampled.resample('M').sum()
                    media_serie_completa = media_serie_completa[media_serie_completa != 0]  
                    media_serie_completa = media_serie_completa.dropna()
                    media_serie_completa = media_serie_completa.mean()
                    #max_serie_completa = df_to_plot_series_resampled.max() 
                    #mmin_serie_completa = df_to_plot_series_resampled.min()    
############################################################################################
                    
                    if k == data_dictionary_keys[0]:
                        
                        header_array = np.array([[key_to_output]])
                        temp_array = np.array([[media_serie_completa]]).T
                        extra_array = np.vstack((header_array,temp_array))
                        
                    elif k == data_dictionary_keys[-1]:
                        temp_array = np.array([[media_serie_completa]]).T
                        extra_array = np.vstack((extra_array,temp_array))  
                        info_array = np.hstack((info_array,extra_array))
                        
                    else:
                        temp_array = np.array([[media_serie_completa]]).T
                        extra_array = np.vstack((extra_array,temp_array))           
                 
                    
                # Escolha o que quer plotado
                if plot_original == True:
                    df_to_plot_series = pd.to_numeric(df_to_plot_series)
                    
                    df_to_plot_series_resampled = df_to_plot_series.resample('D').mean()
                    df_to_plot_series_resampled.plot(kind = 'line',figsize = (20,10), title = title)
                    
                    #df_to_plot_series_resampled = df_to_plot_series.resample('M').mean()
                    #df_to_plot_series_resampled.plot(kind = 'line',figsize = (20,10), title = title)
            
                    
                    #Arte final do grafico
                    plt.grid(which='major', axis='both')
                    plt.grid(which='minor', axis='both')
                            
                    path_to_save = r'D:\estacoes_ba135'
                    comma = k.index("(",0)
                    filename = data_dictionary[k]['nome'][0:comma-1]
                    filename = l + "_" + filename + ".jpg"
                    path_to_save = os.path.join(path_to_save, filename)
                            
                    plt.savefig(path_to_save, figsize = (50,25))
                    plt.show()
                
        np.savetxt("info_array.csv", info_array, delimiter=",",fmt='%s')    
        return info_array
    
def get_series_dadosmensais(climate_df, key, attribute, output = "info_mensal.csv", killzero = False):

    climate_df_keys = list(climate_df.keys())
    for i in climate_df_keys:    
        
        df_to_plot = climate_df[i]['data'].loc['1988-01-01':'2017-12-31']
        df_to_plot = df_to_plot.apply(pd.to_numeric, errors='coerce')

################################ SELECT HERE WHAT DO YOU WANT AN AVERAGE, MAX OR MIN###################
        df_to_plot = df_to_plot.resample('M').mean()


######################################################################################################
        key_to_print = key
        
        for j in range (1,14,1):
            if attribute == 'max':    
                
                if i == climate_df_keys[0]:
                    if j == 1:
                        max_array = np.array([['mes/estacao'], [climate_df[i]['nome']+' - ' + key_to_print]]).T
                        
                    elif j == 13:
                        sep_df = df_to_plot[df_to_plot.index.month.isin([j-1])]
                        sep_series = sep_df[key_to_print].dropna()
                        
                        if killzero==True:
                            sep_series = sep_series[sep_series != 0]  
                            
                        series_max = sep_series.max()
                        value_to_print = str(series_max)
                        max_array_temp = np.vstack([[j-1], [value_to_print]]).T
                        max_array = np.vstack((max_array,max_array_temp))
                        
                    else:
                        sep_df = df_to_plot[df_to_plot.index.month.isin([j-1])]
                        sep_series = sep_df[key_to_print].dropna()
                        
                        if killzero==True:
                            sep_series = sep_series[sep_series != 0]  
                            
                        series_max = sep_series.max()
                        value_to_print = str(series_max)
                        max_array_temp = np.array([[j-1], [value_to_print]]).T
                        max_array = np.vstack((max_array,max_array_temp))
                else:
                    if j == 1:
                        max_array_temp = np.array([[climate_df[i]['nome']+' - ' + key_to_print]]).T

                    elif j == 13 and i != climate_df_keys[-1]:
                        sep_df = df_to_plot[df_to_plot.index.month.isin([j-1])]
                        sep_series = sep_df[key_to_print].dropna()
                        
                        if killzero==True:
                            sep_series = sep_series[sep_series != 0]  
                            
                        series_max = sep_series.max()
                        value_to_print = str(series_max)
                        max_array_temp_temp = np.array([[value_to_print]]).T
                        max_array_temp = np.vstack((max_array_temp,max_array_temp_temp))
                        max_array = np.hstack((max_array,max_array_temp))
                    
                    elif j == 13 and i == climate_df_keys[-1]:
                        sep_df = df_to_plot[df_to_plot.index.month.isin([j-1])]
                        sep_series = sep_df[key_to_print].dropna()
                        series_max = sep_series.max()
                        
                        if killzero==True:
                            sep_series = sep_series[sep_series != 0]  
                            
                        value_to_print = str(series_max)
                        max_array_temp_temp = np.array([[value_to_print]]).T
                        max_array_temp = np.vstack((max_array_temp,max_array_temp_temp))
                        max_array = np.hstack((max_array,max_array_temp))
                        final_array = max_array
                    else:
                        sep_df = df_to_plot[df_to_plot.index.month.isin([j-1])]
                        sep_series = sep_df[key_to_print].dropna()
                        
                        if killzero==True:
                            sep_series = sep_series[sep_series != 0] 
                            
                        series_max = sep_series.max()
                        value_to_print = str(series_max)
                        max_array_temp_temp = np.array([[value_to_print]]).T
                        max_array_temp = np.vstack((max_array_temp,max_array_temp_temp))
           
                             
            elif attribute == 'mean':
                
                if i == climate_df_keys[0]:
                    if j == 1:
                        mean_array = np.array([['mes/estacao'], [climate_df[i]['nome']+' - ' + key_to_print]]).T
                        
                    elif j == 13:
                        sep_df = df_to_plot[df_to_plot.index.month.isin([j-1])]
                        sep_series = sep_df[key_to_print].dropna()
                        
                        if killzero==True:
                            sep_series = sep_series[sep_series != 0]     
                            
                        series_mean = sep_series.mean()
                        value_to_print = str(series_mean)
                        mean_array_temp = np.vstack([[j-1], [value_to_print]]).T
                        mean_array = np.vstack((mean_array,mean_array_temp))
                    else:
                        sep_df = df_to_plot[df_to_plot.index.month.isin([j-1])]
                        sep_series = sep_df[key_to_print].dropna()
                        
                        if killzero==True:
                            sep_series = sep_series[sep_series != 0]   
                            
                        series_mean = sep_series.mean()
                        value_to_print = str(series_mean)
                        mean_array_temp = np.array([[j-1], [value_to_print]]).T
                        mean_array = np.vstack((mean_array,mean_array_temp))
                else:
                    if j == 1:
                        mean_array_temp = np.array([[climate_df[i]['nome']+' - ' + key_to_print]]).T

                    elif j == 13 and i != climate_df_keys[-1]:
                        sep_df = df_to_plot[df_to_plot.index.month.isin([j-1])]
                        sep_series = sep_df[key_to_print].dropna()
                        
                        if killzero==True:
                            sep_series = sep_series[sep_series != 0]   
                            
                        series_mean = sep_series.mean()
                        value_to_print = str(series_mean)
                        mean_array_temp_temp = np.array([[value_to_print]]).T
                        mean_array_temp = np.vstack((mean_array_temp,mean_array_temp_temp))
                        mean_array = np.hstack((mean_array,mean_array_temp))
                        
                    elif j == 13 and i == climate_df_keys[-1]:
                        sep_df = df_to_plot[df_to_plot.index.month.isin([j-1])]
                        sep_series = sep_df[key_to_print].dropna()
                        
                        if killzero==True:
                            sep_series = sep_series[sep_series != 0]   
                        
                        series_mean = sep_series.mean()
                        value_to_print = str(series_mean)
                        mean_array_temp_temp = np.array([[value_to_print]]).T
                        mean_array_temp = np.vstack((mean_array_temp,mean_array_temp_temp))
                        mean_array = np.hstack((mean_array,mean_array_temp))
                        final_array = mean_array 
                    else:
                        sep_df = df_to_plot[df_to_plot.index.month.isin([j-1])]
                        sep_series = sep_df[key_to_print].dropna()

                        if killzero==True:
                            sep_series = sep_series[sep_series != 0]   

                        series_mean = sep_series.mean()
                        value_to_print = str(series_mean)
                        mean_array_temp_temp = np.array([[value_to_print]]).T
                        mean_array_temp = np.vstack((mean_array_temp,mean_array_temp_temp))
                
            
            elif attribute == 'min':
                
                if i == climate_df_keys[0]:
                    
                    if j == 1:
                        min_array = np.array([['mes/estacao'], [climate_df[i]['nome']+' - ' + key_to_print]]).T
                        
                    elif j == 13:
                        sep_df = df_to_plot[df_to_plot.index.month.isin([j-1])]
                        sep_series = sep_df[key_to_print].dropna()
                        
                        if killzero==True:
                            sep_series = sep_series[sep_series != 0]   

                        series_min = sep_series.min()
                        value_to_print = str(series_min)
                        min_array_temp = np.vstack([[j-1], [value_to_print]]).T
                        min_array = np.vstack((min_array,min_array_temp))
                    else:
                        series_min = df_to_plot[df_to_plot.index.month.isin([j-1])].min()
                        value_to_print = str(series_min[key_to_print])
                        
                        min_array_temp = np.array([[j-1], [value_to_print]]).T
                        min_array = np.vstack((min_array,min_array_temp))
                else:
                    if j == 1:
                        min_array_temp = np.array([[climate_df[i]['nome']+' - ' + key_to_print]]).T

                    elif j == 13:
                        sep_df = df_to_plot[df_to_plot.index.month.isin([j-1])]
                        sep_series = sep_df[key_to_print].dropna()
                        
                        if killzero==True:
                            sep_series = sep_series[sep_series != 0]   

                        series_min = sep_series.min()
                        value_to_print = str(series_min)
                        min_array_temp_temp = np.array([[value_to_print]]).T
                        min_array_temp = np.vstack((min_array_temp,min_array_temp_temp))
                        min_array = np.hstack((min_array,min_array_temp))
                        final_array = min_array
                    else:
                        sep_df = df_to_plot[df_to_plot.index.month.isin([j-1])]
                        sep_series = sep_df[key_to_print].dropna()
                                                
                        if killzero==True:
                            sep_series = sep_series[sep_series != 0]   

                        series_min = sep_series.min()
                        value_to_print = str(series_min)
                        min_array_temp_temp = np.array([[value_to_print]]).T
                        min_array_temp = np.vstack((min_array_temp,min_array_temp_temp))
    
    np.savetxt(output, final_array, delimiter=",",fmt='%s')    
    print(final_array)
    if output == "info_mensal.csv":
        print("""Seu arquivo de saída está na pasta de trabalho do Python para 
          este projeto. Use os.chdir para mudar. O arquivo está nomeado
          "info_mensal.csv". """)


def printa_serie_dadosmensais(arquivo, root, nplot):
    
    array = np.loadtxt(arquivo, dtype=str, delimiter=',')
    counter = 0
    colors = ['red', 'navy', 'cyan','magenta','gray']
    colors_single = ['#4285F4', '#FBBC05', '#34A853', '#EA4335',]
    fig, ax = plt.subplots(figsize=(17,8))
    if nplot == 1:
        
        for t in range(1,array.shape[1],1):
            
            date_array = np.arange(1,13,1)
            serie = array[1:13,t:t+1].astype(float)
            header = array[0,t]
            xticks = ["J","F","M","A","M","J","J","A","S","O","N","D"]
            #yticks = np.linspace(0,1100,12)
            media = serie.astype(np.float).mean()
            media_y = np.array(np.zeros(12)+media)
            media_x = np.arange(1,13,1)
                    
            x_smooth = np.linspace(1,12,300)
            y_smooth = spline(media_x, serie, x_smooth)
            y_smooth = y_smooth[:,0]
            
            ax.set_ylabel('Temperatura '+ '($^\circ$C)', fontsize=26)
            ax.set_xlabel('Meses do ano', fontsize=26)
            #ax.set_yticks(yticks)
            
            #ax.plot(media_x, media_y, color = "orange", lw=1.5, linestyle = '-')
            #ax.scatter(date_array, final_array[:,0], color = "red")
            #ax.annotate('Tm = %.1f ' % media + r'$^\circ$C', xy=(5, media), xytext=(5, (media+0.2)),  fontsize=27)
            
            
            ax.plot(media_x, serie, label= 'Temp. Méd. Comp-%s'%header, color = colors_single[counter], lw=2.5, alpha = 0.8)
            plt.scatter(media_x, serie, s=100, color = colors_single[counter])
            counter = counter+1
            
        ax.set_xticklabels(xticks)
        ax.set_xlim(xmin=1, xmax=12)
        ax.set_ylim(15,45)
        ax.set_xticks(date_array)
        ax.xaxis.set_tick_params(labelsize=26)
        ax.yaxis.set_tick_params(labelsize=26)
        ax.yaxis.set_label_coords(-0.03,0.5)
        plt.grid(which='both', axis='y')
        ax.spines['bottom'].set_color('0.5')
        ax.spines['top'].set_color('0.5')
        ax.spines['right'].set_color('0.5')
        ax.spines['left'].set_color('0.5')
        
        plt.tight_layout()
        plt.grid(True, axis = 'x')
        plt.grid(True, axis = 'y')
        plt.legend(loc = 'best', fontsize = 22)
        #plt.ylim(0,1100)
    
        ax.margins(x=0)
        #plt.title("Temperaturas mínimas mensais - %s" % (header), y=1.03, fontsize = 30)
        filetarget = (os.path.join(root, "\%s_plot.png"% header))
        plt.savefig(filetarget,dpi = 300)
        plt.show()
        
    else:
    
        for t in range(1,int((array.shape[1]-1)/2)+1,1):
            
            nseries = int((array.shape[1] - 1)/2)
            date_array = np.arange(1,13,1)
            serie1 = array[1:13,t:t+1].astype(float)
            serie2 = array[1:13,t+nseries:t+nseries+1].astype(float)
            
            header = array[0,t]
            xticks = ["J","F","M","A","M","J","J","A","S","O","N","D"]
            #yticks = np.linspace(0,1100,12)
            media_serie1 = serie1.astype(np.float).mean()
            media_serie2 = serie2.astype(np.float).mean()
            media_y_serie1 = np.array(np.zeros(12)+media_serie1)
            media_y_serie2 = np.array(np.zeros(12)+media_serie2)
            media_x = np.arange(1,13,1)
        
            fig, ax = plt.subplots(figsize=(17,8))
            
            #x_smooth = np.linspace(1,12,300)
            #y_smooth_serie1 = spline(media_x, serie1, x_smooth)
            #y_smooth_serie2 = spline(media_x, serie2, x_smooth)
            #y_smooth_serie1 = y_smooth_serie1[:,0]
            #y_smooth_serie2 = y_smooth_serie2[:,0]
            
            #ax.fill_between(x_smooth, y_smooth, cmap = "Blues", color = 'darkblue',alpha = 0.8)
            ax.set_ylabel('Temperatura '+ '($^\circ$C)', fontsize=26)
            ax.set_xlabel('Meses do ano', fontsize=26)
            #ax.set_yticks(yticks)
            
            #ax.plot(x_smooth, y_smooth_serie1, color = "navy", lw=1.5)
            #ax.plot(x_smooth, y_smooth_serie2, color = "navy", lw=1.5)
            
            ax.plot(media_x, serie1, label= 'Mín.-Temperaturas Mínimas', color = "red", lw=2.5)
            ax.plot(media_x, serie2, label= 'Méd.-Temperaturas Mínimas', color = "navy", lw=2.5)
            plt.scatter(media_x, serie1, s=100, color = "red", alpha = 0.8)
            plt.scatter(media_x, serie2, s=100, color = "navy", alpha = 0.8)
            #ax.scatter(date_array, final_array[:,0], color = "red")
                       
            ax.set_xticklabels(xticks)
            ax.set_xlim(xmin=1, xmax=12)
            ax.set_ylim(0,45)
            ax.set_xticks(date_array)
            ax.xaxis.set_tick_params(labelsize=23)
            ax.yaxis.set_tick_params(labelsize=23)
            ax.yaxis.set_label_coords(-0.03,0.5)
            plt.grid(which='both', axis='y')
            ax.spines['bottom'].set_color('0.5')
            ax.spines['top'].set_color('0.5')
            ax.spines['right'].set_color('0.5')
            ax.spines['left'].set_color('0.5')
            
            plt.tight_layout()
            plt.grid(True, axis = 'x')
            plt.grid(True, axis = 'y')
            plt.legend(loc = 'lower left', fontsize = 22)
            #plt.ylim(0,1100)
        
            ax.margins(x=0)
            #plt.title("Temperaturas mínimas mensais - %s" % (header), y=1.03, fontsize = 30)
            filetarget = (os.path.join(root, "\%s_plot.png"% header))
            plt.savefig(filetarget,dpi = 300)
            plt.show()

def printa_arquivoevap(arquivo):
    
    array = np.loadtxt(arquivo, dtype=str, delimiter=',')
    counter = 0
    colors = ['red', 'navy', 'cyan','magenta','gray']
    fig, ax = plt.subplots(figsize=(17,8))
    
    for t in range(1,array.shape[1],1):
    
        date_array = np.arange(1,13,1)
        serie = array[1:13,t:t+1].astype(float)
        header = array[0,t]
        xticks = ["J","F","M","A","M","J","J","A","S","O","N","D"]
        #yticks = np.linspace(0,1100,12)
        media = serie.astype(np.float).mean()
        media_y = np.array(np.zeros(12)+media)
        media_x = np.arange(1,13,1)
                
        x_smooth = np.linspace(1,12,300)
        y_smooth = spline(media_x, serie, x_smooth)
        y_smooth = y_smooth[:,0]
        
        #ax.fill_between(x_smooth, y_smooth, cmap = "Blues", color = 'darkblue',alpha = 0.8)
        ax.set_ylabel('Temperatura'+ '$^\circ$C', fontsize=26)
        ax.set_xlabel('Meses do ano', fontsize=26)
        #ax.set_yticks(yticks)
        
        #ax.plot(media_x, media_y, color = "orange", lw=1.5, linestyle = '-')
        #ax.scatter(date_array, final_array[:,0], color = "red")
        #ax.annotate('Tm = %.1f ' % media + r'$^\circ$C', xy=(5, media), xytext=(5, (media+0.2)),  fontsize=27)
        
        counter = counter+1
        ax.plot(media_x, serie, label= 'Temperaturas Médias compensadas - %s($^\circ$C)'%header, color = colors[counter], lw=2.5, alpha = 0.8)
        plt.scatter(media_x, serie, s=100, color = colors[counter])
       
    ax.set_xticklabels(xticks)
    ax.set_xlim(xmin=1, xmax=12)
    ax.set_ylim(15,45)
    ax.set_xticks(date_array)
    ax.xaxis.set_tick_params(labelsize=23)
    ax.yaxis.set_tick_params(labelsize=23)
    ax.yaxis.set_label_coords(-0.03,0.5)
    plt.grid(which='both', axis='y')
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')
    
    plt.tight_layout()
    plt.grid(True, axis = 'x')
    plt.grid(True, axis = 'y')
    plt.legend(loc = 'best', fontsize = 22)
    #plt.ylim(0,1100)

    ax.margins(x=0)
    #plt.title("Temperaturas mínimas mensais - %s" % (header), y=1.03, fontsize = 30)
    filetarget = (os.path.join(root, "\%s_plot.png"% header))
    plt.savefig(filetarget,dpi = 300)
    plt.show()
    
def printavento(arquivo,root):
    
    array = np.loadtxt(arquivo, dtype=str, delimiter=',')
    counter = 0
    colors_bar = ['#4285F4', '#FBBC05', '#34A853', '#EA4335',]
    colors_scatter = ['#4285F4', '#FBBC05', '#34A853', '#EA4335',]
    colors_lines = ['#4285F4', '#FBBC05', '#34A853', '#EA4335',]
                      
    fig, ax = plt.subplots(figsize=(17,8))
    
    ncols = int((array.shape[1]-1)/2)
    
    for t in range(1,ncols+1,1):
        
        nseries = int((array.shape[1] - 1)/2)
        date_array = np.arange(1,13,1)
        serie1 = array[1:13,t:t+1].astype(float)
        serie2 = array[1:13,t+nseries:t+nseries+1].astype(float)
        
        header = array[0,t]
        xticks = ["J","F","M","A","M","J","J","A","S","O","N","D"]
        #yticks = np.linspace(0,1100,12)
        media_x = list(range(1,13,1))
        media_x = np.array([media_x]).T
            
        width = 0.15
        positions = [-0.3, -0.15,0,0.15]
        
        ax.set_ylabel('Velocidade do Vento '+ '(m/s)', fontsize=26)
        ax.set_xlabel('Meses do ano', fontsize=26)

        bar_x = media_x+positions[counter]
      
        
        ax.plot(media_x, serie1, label= 'Vel. Máx do Vento'+ '-%s'%header, lw=2.5,
                color = colors_lines[counter])
        
        
        ax.bar(bar_x, serie2, width = 0.15, color = colors_bar[counter], 
               label= 'Vel. Média do Vento'+ '-%s'%header, lw=2.5,zorder=2, 
               alpha = 0.4, edgecolor='black')
        ax.bar(bar_x, serie2, width = 0.15, color = 'white', 
               lw=1,zorder=1, alpha = 1, 
               edgecolor='black')
        
        plt.scatter(media_x, serie1, s=40, color = colors_scatter[counter], zorder=2)
        counter+=1
                   
    ax.set_xticklabels(xticks)
    ax.set_xlim(xmin=0.20, xmax=12.50)
    ax.set_ylim(0,5)
    ax.set_xticks(date_array)
    #ax.yaxis.tick_right()
    ax.xaxis.set_tick_params(labelsize=23)
    ax.yaxis.set_tick_params(labelsize=23)
    ax.yaxis.set_label_coords(-0.03,0.5)
    plt.grid(which='both', axis='y')
    ax.spines['bottom'].set_color('0.5')
    ax.spines['top'].set_color('0.5')
    ax.spines['right'].set_color('0.5')
    ax.spines['left'].set_color('0.5')
    
    plt.tight_layout()
    plt.grid(True, axis = 'x')
    plt.grid(True, axis = 'y')
    plt.legend(loc = 'best', ncol=2, fontsize = 18)

    ax.margins(x=0)
    filetarget = (os.path.join(root, "vento_plot.png"))
    plt.savefig(filetarget,dpi = 300)
    plt.show()
    

def processa_dfevap(data_dictionary, output):

    print('Comecando a printar dado para mapa...')
    data_df_keys = [' BARREIRAS - BA (OMM: 83236)',
                    ' CORRENTINA - BA (OMM: 83286)',  
                    ' STa  R  DE CASSIA  IBIPETUBA  - BA (OMM: 83076)',
                    ' TAGUATINGA - TO (OMM: 83235)']
    #colors_bar = ['#d5e1df', '#e3eaa7', '#b5e7a0','#86af49','#405d27']
    colors_lines = ['#4285F4', '#FBBC05', '#34A853', '#EA4335',]
    colors_bar =  ['#4285F4', '#FBBC05', '#34A853', '#EA4335',]
    #colors_lines = ['red', 'blue', 'orange', 'green']
    
    width = 0.15
    positions = [-0.3, -0.15,0,0.15]
    fig, ax1 = plt.subplots(figsize=(17,8))

    counter = 0
    for k in data_df_keys:
        
        data_dictionary[k]['data'] = data_dictionary[k]['data'].loc['1988-01-01':'2017-12-31']
        xticks = ["J","F","M","A","M","J","J","A","S","O","N","D"]
        df_to_plot = data_dictionary[k]['data'].dropna()
        df_to_plot = df_to_plot.apply(pd.to_numeric, errors='coerce')

        evap_temp = np.zeros((12,1))
        tmedia_temp = np.zeros((12,1))
    
        date_array = np.arange(1,13,1)
        
        headers = ['Barreiras', 'Correntina', 'Sta. Rita de Cássia-Ibipetuba','Taguatinga']
        
        evap = df_to_plot['Evaporacao Piche']
        evap = evap.resample('M').sum()
        tmedia = df_to_plot['Temp Comp Media']
        tmedia = tmedia.resample('D').mean()
    
        for m in list(range(1,13)):

            evap_temp_temp = evap[evap.index.month.isin([m])]
            evap_temp_temp = evap_temp_temp.dropna()
            evap_temp_temp = evap_temp_temp[evap_temp_temp!=0]
            evap_mean = evap_temp_temp.mean()
                       
            tmedia_temp_temp = tmedia[tmedia.index.month.isin([m])]
            tmedia_temp_temp = tmedia_temp_temp.dropna()
            tmedia_temp_temp = tmedia_temp_temp[tmedia_temp_temp !=0]
            tmedia_mean = tmedia_temp_temp.mean()
            
            evap_temp[m-1,0] = evap_mean
            tmedia_temp[m-1,0] = tmedia_mean
            
        #yticks = np.linspace(0,1100,12)
        
        bar_x = date_array+positions[counter]
        bar_x = np.array([bar_x]).T
        
        ax1.bar(bar_x, evap_temp, width = 0.15, color = colors_bar[counter], 
               label= 'Evap. Méd. Piche'+ '-%s'%headers[counter], lw=2.5,zorder=1)
        
        ax1.set_ylabel('Evaporação média diária '+ '(mm)', fontsize=26)
        ax1.set_xlabel('Meses do ano', fontsize=26)                   
        ax1.set_xticklabels(xticks)
        ax1.set_xlim(xmin=0.20, xmax=12.50)
        ax1.set_ylim(0,600)
        ax1.set_xticks(date_array)
        ax1.xaxis.set_tick_params(labelsize=23)
        ax1.yaxis.set_tick_params(labelsize=23)
        ax1.yaxis.set_label_coords(-0.05,0.5)
        ax1.spines['bottom'].set_color('0.5')
        ax1.spines['top'].set_color('0.5')
        ax1.spines['right'].set_color('0.5')
        ax1.spines['left'].set_color('0.5')
        ax1.margins(x=0)
        ax1.legend(loc = 'upper right', ncol=1, fontsize = 14) 
        ax1.grid(True, axis = 'x')
        #ax1.grid(True, axis = 'y')
        
        if k == data_df_keys[0]:
            ax2 = ax1.twinx()

        ax2.plot(date_array, tmedia_temp, label= 'Temp. Méd. Comp.'+ '-%s'%headers[counter], lw=2.5,
        color = colors_lines[counter])
        ax2.scatter(date_array, tmedia_temp, s=40, color = colors_lines[counter], zorder=2)
        counter+=1
        ax2.legend(loc = 'upper left', ncol=1, fontsize = 14) 
        ax2.set_ylabel('Temp Comp. Média  '+ '($^\circ$C)', fontsize=26)
        ax2.set_ylim(0,35)
        ax2.yaxis.set_tick_params(labelsize=23)
            
    plt.tight_layout()
    plt.grid(True, axis = 'x')
    plt.grid(True, axis = 'y')
    
    if output == "evapo_plot.png":
        filetarget = (os.path.join(root, "evapo_plot.png"))
    
    plt.savefig(filetarget,dpi = 300)
    plt.show()

def processavento(climate_df):
    
    climate_df_keys = list(climate_df.keys())
   
    for i in climate_df_keys:    

#Creates the first frame 
        if i == climate_df_keys[0]:
            counter = 0
            df_to_plot = climate_df[i]['data']
            station_name = climate_df[i]['nome']
            df_to_plot = df_to_plot.apply(pd.to_numeric, errors='coerce')
            #df_to_plot = df_to_plot.resample('D').mean()
            df_to_plot = df_to_plot.loc['1988-01-01':'2017-12-31']
            df_to_plot = df_to_plot.dropna()
            unique = df_to_plot.DirecaoVento.unique()
            
            for u in unique: 
            
                temp_df = df_to_plot[df_to_plot['DirecaoVento'] == unique[counter]]
                nregistros = df_to_plot.shape
                media = temp_df['VelocidadeVentoMedia'].mean()
                maxima = temp_df['VelocidadeVentoMedia'].max()
                z = temp_df['DirecaoVento'].value_counts()
                z = z.values[0]
                frequencia = z/nregistros[0]               
                counter+=1
            
                if u ==unique[0]:
                    index = np.array([['estacao'], ['atributo'], [u]])
                    vento_station_header = np.array([[station_name], [station_name], [station_name]]).T
                    vento_header = np.array([['maxima'], ['media'], ['freq']]).T
                    vento = np.array([[maxima], [media], [frequencia]]).T
                    vento = np.vstack((vento_header, vento))
                    vento = np.vstack((vento_station_header, vento))
                    vento = np.hstack((index,vento))
                
                else:
                    vento_temp = np.array([[u],[maxima], [media], [frequencia]]).T
                    vento = np.vstack((vento, vento_temp))
# Creates non-first frames    
        else:
            counter = 0
            df_to_plot = climate_df[i]['data']
            station_name = climate_df[i]['nome']
            df_to_plot = df_to_plot.apply(pd.to_numeric, errors='coerce')
            #df_to_plot = df_to_plot.resample('D').mean()
            df_to_plot = df_to_plot.loc['1988-01-01':'2017-12-31']
            df_to_plot = df_to_plot.dropna()
            unique = df_to_plot.DirecaoVento.unique()
            
            for v in unique: 
                
                temp_df = df_to_plot[df_to_plot['DirecaoVento'] == unique[counter]]
                nregistros = df_to_plot.shape
                media = temp_df['VelocidadeVentoMedia'].mean()
                maxima = temp_df['VelocidadeVentoMedia'].max()
                z = temp_df['DirecaoVento'].value_counts()
                z = z.values[0]
                frequencia = z/nregistros[0]               
                counter+=1
            
                if v ==unique[0]:   
                    current_index = vento[2:,0]
                    v_str = '%.1f' % v
                    if v_str in current_index:
                        length_index = current_index.shape[0]
                        row_to_change = np.where(current_index == v_str)[0][0]+2
                        vento_temp = np.zeros((length_index, 3))
                        vento_station_header = np.array([[station_name], [station_name], [station_name]]).T
                        vento_header = np.array([['maxima'], ['media'], ['freq']]).T
                        vento_temp = np.vstack((vento_header, vento_temp))
                        vento_temp = np.vstack((vento_station_header, vento_temp))
                        
                        vento_temp[row_to_change,0] = maxima 
                        vento_temp[row_to_change,1] = media 
                        vento_temp[row_to_change,2] = frequencia
                        
                    else: 
                        exit()
                elif v!=unique[0] and v!=unique[-1]:
                    current_index = vento[2:,0]
                    v_str = '%.1f' % v
                    if v_str in current_index:
                        length_index = current_index.shape[0]
                        row_to_change = np.where(current_index == v_str)[0][0]+2
                                                
                        vento_temp[row_to_change,0] = maxima 
                        vento_temp[row_to_change,1] = media 
                        vento_temp[row_to_change,2] = frequencia
                    
                    else: 
                        current_ncols = vento.shape[1]
                        blank = np.zeros((1,current_ncols))
                        vento = np.vstack((vento, blank))
                        lines = vento.shape[0]
                        vento[lines-1, 0] = v_str
                        
                        current_index = vento[2:,0]
                        
                        blank_temp = np.zeros((1,3))
                        vento_temp = np.vstack((vento_temp, blank_temp))
                        row_to_change = np.where(current_index == v_str)[0][0]+2
                        
                        vento_temp[row_to_change,0] = maxima 
                        vento_temp[row_to_change,1] = media 
                        vento_temp[row_to_change,2] = frequencia                       
                        
                
                elif v!=unique[0] and v==unique[-1]:
                    if v_str in current_index:
                        length_index = current_index.shape[0]
                        row_to_change = np.where(current_index == v_str)[0][0]+2
                            
                        vento_temp[row_to_change,0] = maxima 
                        vento_temp[row_to_change,1] = media 
                        vento_temp[row_to_change,2] = frequencia
                    
                    else: 
                        current_ncols = vento.shape[1]
                        blank = np.zeros((1,current_ncols))
                        vento = np.vstack((vento, blank))
                        lines = vento.shape[0]
                        vento[lines-1, 0] = v_str
                        
                        current_index = vento[2:,0]
                        
                        blank_temp = np.zeros((1,3))
                        vento_temp = np.vstack((vento_temp, blank_temp))
                        vento_temp = np.vstack((vento_header, vento_temp))
                        row_to_change = np.where(current_index == v_str)
                        
                        vento_temp[row_to_change[0][0]-3,0] = maxima 
                        vento_temp[row_to_change[0][0]-3,1] = media 
                        vento_temp[row_to_change[0][0]-3,2] = frequencia
                    
                                       
                    vento = np.hstack((vento,vento_temp))
        
    np.savetxt("vento.csv", vento, delimiter=",",fmt='%s')    
    print(vento)
    
def processainsolacao(data_dictionary):

    print('Comecando a printar dado para mapa...')
    data_df_keys = [' BARREIRAS - BA (OMM: 83236)',
                    ' CORRENTINA - BA (OMM: 83286)',  
                    ' STa  R  DE CASSIA  IBIPETUBA  - BA (OMM: 83076)',
                    ' TAGUATINGA - TO (OMM: 83235)']
    colors_scatter = ['#4285F4', '#FBBC05', '#34A853', '#EA4335',]
    colors_lines = ['#4285F4', '#FBBC05', '#34A853', '#EA4335',]
    
    width = 0.15
    positions = [-0.3, -0.15,0,0.15]
    fig, ax1 = plt.subplots(figsize=(17,8))

    counter = 0
    for k in data_df_keys:
        
        data_dictionary[k]['data'] = data_dictionary[k]['data'].loc['1988-01-01':'2017-12-31']
        xticks = ["J","F","M","A","M","J","J","A","S","O","N","D"]
        df_to_plot = data_dictionary[k]['data']
        df_to_plot = df_to_plot.apply(pd.to_numeric, errors='coerce')

        insolacao_temp = np.zeros((12,1))
    
        date_array = np.arange(1,13,1)
        
        headers = ['Barreiras', 'Correntina', 'Sta. Rita de Cássia-Ibipetuba','Taguatinga']
        
        insolacao = df_to_plot['Insolacao']
    
        for m in list(range(1,13)):

            insolacao_temp_temp = insolacao.resample('M').sum()
            insolacao_temp_temp = insolacao_temp_temp[insolacao_temp_temp.index.month.isin([m])]
            insolacao_temp_temp = insolacao_temp_temp.dropna()
            insolacao_temp_temp = insolacao_temp_temp.mean()
             #sep_df = df_to_plot[df_to_plot.index.month.isin([j-1])]
            
            insolacao_temp[m-1,0] = insolacao_temp_temp           
        #yticks = np.linspace(0,1100,12)
        
        bar_x = date_array+positions[counter]
        bar_x = np.array([bar_x]).T
        
        ax1.plot(date_array, insolacao_temp, label= 'Insolação'+ '-%s'%headers[counter], lw=2.5,
        color = colors_lines[counter])
        
        ax1.scatter(date_array, insolacao_temp, s=40, color = colors_lines[counter], zorder=2)
        counter+=1
        ax1.set_ylabel('Insolação Mensal'+ '(h)', fontsize=26)
        ax1.set_xlabel('Meses do ano', fontsize=26)                   
        ax1.set_xticklabels(xticks)
        ax1.set_xlim(xmin=0.97, xmax=12.03)
        ax1.set_ylim(0,500)
        ax1.set_xticks(date_array)
        ax1.xaxis.set_tick_params(labelsize=23)
        ax1.yaxis.set_tick_params(labelsize=23)
        ax1.yaxis.set_label_coords(-0.05,0.5)
        ax1.spines['bottom'].set_color('0.5')
        ax1.spines['top'].set_color('0.5')
        ax1.spines['right'].set_color('0.5')
        ax1.spines['left'].set_color('0.5')
        ax1.margins(x=0)
        ax1.legend(loc = 'upper right', ncol=2, fontsize = 18) 
        ax1.grid(True, axis = 'x')
        #ax1.grid(True, axis = 'y')
                  
    plt.tight_layout()
    plt.grid(True, axis = 'x')
    plt.grid(True, axis = 'y')
    
    filetarget = (os.path.join(root, "termo_plot.png"))
    plt.savefig(filetarget,dpi = 300)
    plt.show()
    

def printa_termopluviometrico(climate_df):
    counter = 0
    colors_bar = ['#4285F4', '#FBBC05', '#34A853', '#EA4335',]
    colors_scatter = ['#4285F4', '#FBBC05', '#34A853', '#EA4335',]
    colors_lines = ['#4285F4', '#FBBC05', '#34A853', '#EA4335',]
                      
    fig, ax1 = plt.subplots(figsize=(17,8))
    data_df_keys = list(climate_df.keys())

    date_array = np.arange(1,13,1)
    width = 0.15
    positions = [-0.3, -0.15,0,0.15]
    
    headers = ['Barreiras', 'Correntina', 'Sta. Rita de Cassia-Ibipetuba','Taguatinga']
    
    data_df_keys = [' BARREIRAS - BA (OMM: 83236)',
                    ' CORRENTINA - BA (OMM: 83286)',  
                    ' STa  R  DE CASSIA  IBIPETUBA  - BA (OMM: 83076)',
                    ' TAGUATINGA - TO (OMM: 83235)']
    for i in data_df_keys:
        df_to_plot = climate_df[i]['data']
        df_to_plot = df_to_plot.apply(pd.to_numeric, errors='coerce')
        precipitacao_temp = np.zeros((12,1))
        tmedia_temp = np.zeros((12,1))
        
        precipitacao = df_to_plot['Precipitacao'].resample('M').sum()
        tmedia = df_to_plot['Temp Comp Media'].resample('D').mean()
        
        for m in list(range(1,13)):

            precipitacao_temp_temp = precipitacao[precipitacao.index.month.isin([m])]
            precipitacao_temp_temp = precipitacao_temp_temp.dropna()
            precipitacao_temp_temp = precipitacao_temp_temp[precipitacao_temp_temp != 0] 
            precipitacao_mean = precipitacao_temp_temp.mean()
            
            tmedia_temp_temp = tmedia[tmedia.index.month.isin([m])]
            tmedia_temp_temp = tmedia_temp_temp.dropna()
            tmedia_temp_temp = tmedia_temp_temp[tmedia_temp_temp != 0] 
            tmedia_mean = tmedia_temp_temp.mean()
            
            precipitacao_temp[m-1,0] = precipitacao_mean
            tmedia_temp[m-1,0] = tmedia_mean
                                                 
        xticks = ["J","F","M","A","M","J","J","A","S","O","N","D"]
        #yticks = np.linspace(0,1100,12)
        
        bar_x = date_array+positions[counter]
        bar_x = np.array([bar_x]).T
        
        ax1.bar(bar_x, precipitacao_temp, width = 0.15, color = colors_bar[counter], 
               label= 'Precipitacao Méd. Mensal'+ '-%s'%headers[counter], lw=2.5,zorder=1)
        
        ax1.set_ylabel('Precipitacao Mensal '+ '(mm)', fontsize=26)
        ax1.set_xlabel('Meses do ano', fontsize=26)                   
        ax1.set_xticklabels(xticks)
        ax1.set_xlim(xmin=0.20, xmax=12.50)
        ax1.set_ylim(0,450)
        ax1.set_xticks(date_array)
        ax1.xaxis.set_tick_params(labelsize=23)
        ax1.yaxis.set_tick_params(labelsize=23)
        ax1.yaxis.set_label_coords(-0.05,0.5)
        ax1.spines['bottom'].set_color('0.5')
        ax1.spines['top'].set_color('0.5')
        ax1.spines['right'].set_color('0.5')
        ax1.spines['left'].set_color('0.5')
        ax1.margins(x=0)
        ax1.legend(loc = 'upper right', ncol=1, fontsize = 14) 
        ax1.grid(True, axis = 'x')
        #ax1.grid(True, axis = 'y')
        
        if i == data_df_keys[0]:
            ax2 = ax1.twinx()

        ax2.plot(date_array, tmedia_temp, label= 'Temperatura'+ '-%s'%headers[counter], lw=2.5,
        color = colors_lines[counter])
        ax2.scatter(date_array, tmedia_temp, s=40, color = colors_lines[counter], zorder=2)
        
        ax2.legend(loc = 'upper left', ncol=1, fontsize = 14) 
        ax2.set_ylim(0,35)
        ax2.set_ylabel('Temp. Méd Comp. Diária '+ '($^\circ$C)', fontsize=26)
        ax2.yaxis.set_tick_params(labelsize=23)
        
        counter+=1
        
            
    plt.tight_layout()
    plt.grid(True, axis = 'x')
    plt.grid(True, axis = 'y')
    
    filetarget = (os.path.join(root, "termo_plot.png"))
    plt.savefig(filetarget,dpi = 300)
    plt.show()
    
def printa_termopluvioevaporimetrico(climate_df):
    counter = 0
    colors_bar = ['#4285F4', '#FBBC05', '#34A853', '#EA4335',]
    colors_scatter = ['#4285F4', '#FBBC05', '#34A853', '#EA4335',]
    colors_lines = ['#4285F4', '#FBBC05', '#34A853', '#EA4335',]
                      
    fig, ax1 = plt.subplots(figsize=(26.7,15), dpi=300)
    data_df_keys = list(climate_df.keys())

    date_array = np.arange(1,13,1)
    width = 0.15
    positions = [-0.3, -0.15,0,0.15]
    
    headers = ['Barreiras', 'Correntina', 'Sta. Rita de Cassia-Ibipetuba','Taguatinga']
    
    data_df_keys = [' BARREIRAS - BA (OMM: 83236)',
                    ' CORRENTINA - BA (OMM: 83286)',  
                    ' STa  R  DE CASSIA  IBIPETUBA  - BA (OMM: 83076)',
                    ' TAGUATINGA - TO (OMM: 83235)']
    for i in data_df_keys:
        df_to_plot = climate_df[i]['data']
        df_to_plot = df_to_plot.apply(pd.to_numeric, errors='coerce')
        precipitacao_temp = np.zeros((12,1))
        tmedia_temp = np.zeros((12,1))
        evap_temp = np.zeros((12,1))
        
        precipitacao = df_to_plot['Precipitacao'].resample('M').sum()
        tmedia = df_to_plot['Temp Comp Media'].resample('D').mean()
        
        evap = df_to_plot['Evaporacao Piche'].resample('M').sum()
        
        for m in list(range(1,13)):

            precipitacao_temp_temp = precipitacao[precipitacao.index.month.isin([m])]
            precipitacao_temp_temp = precipitacao_temp_temp.dropna()
            precipitacao_temp_temp = precipitacao_temp_temp[precipitacao_temp_temp != 0] 
            precipitacao_mean = precipitacao_temp_temp.mean()
            
            tmedia_temp_temp = tmedia[tmedia.index.month.isin([m])]
            tmedia_temp_temp = tmedia_temp_temp.dropna()
            tmedia_temp_temp = tmedia_temp_temp[tmedia_temp_temp != 0] 
            tmedia_mean = tmedia_temp_temp.mean()
            
            evap_temp_temp = evap[evap.index.month.isin([m])]
            evap_temp_temp = evap_temp_temp.dropna()
            evap_temp_temp = evap_temp_temp[evap_temp_temp!=0]
            evap_mean = evap_temp_temp.mean()
            
            precipitacao_temp[m-1,0] = precipitacao_mean
            tmedia_temp[m-1,0] = tmedia_mean
            evap_temp[m-1,0] = evap_mean
                                                 
        xticks = ["J","F","M","A","M","J","J","A","S","O","N","D"]
        #yticks = np.linspace(0,1100,12)
        
        bar_x = date_array+positions[counter]
        bar_x = np.array([bar_x]).T
        
        ax1.bar(bar_x, precipitacao_temp, width = 0.15, color = colors_bar[counter], 
               label= 'Precipitação Méd. Mensal'+ '-%s'%headers[counter], lw=2.5,
               alpha = 0.5, zorder=2)
        
        ax1.bar(bar_x, precipitacao_temp, width = 0.15, color = 'white', 
               lw=1,zorder=1, alpha = 1, 
               edgecolor='black')
        
        ax1.plot(date_array, evap_temp, label= 'Evaporação'+ '-%s'%headers[counter], lw=2.5,
        color = colors_lines[counter], linestyle = '-.')
        
        ax1.scatter(date_array, evap_temp, s=40, color = colors_lines[counter], zorder=2, 
                    )
        
        ax1.set_ylabel('Precipitação/Evaporação '+ '(mm)', fontsize=26)
        ax1.set_xlabel('Meses do ano', fontsize=26)                   
        ax1.set_xticklabels(xticks)
        ax1.set_xlim(xmin=0.20, xmax=12.50)
        ax1.set_ylim(0,350)
        ax1.set_xticks(date_array)
        ax1.xaxis.set_tick_params(labelsize=23)
        ax1.yaxis.set_tick_params(labelsize=23)
        ax1.yaxis.set_label_coords(-0.05,0.5)
        ax1.spines['bottom'].set_color('0.5')
        ax1.spines['top'].set_color('0.5')
        ax1.spines['right'].set_color('0.5')
        ax1.spines['left'].set_color('0.5')
        ax1.margins(x=0)
        ax1.legend(loc = 'upper right', ncol=1, fontsize = 14) 
        ax1.grid(True, axis = 'x')
        #ax1.grid(True, axis = 'y')
        
        if i == data_df_keys[0]:
            ax2 = ax1.twinx()

        ax2.plot(date_array, tmedia_temp, label= 'Temperatura'+ '-%s'%headers[counter], lw=2.5,
        color = colors_lines[counter])
        ax2.scatter(date_array, tmedia_temp, s=40, color = colors_lines[counter], zorder=2)
        
        ax2.legend(loc = 'upper left', ncol=1, fontsize = 14) 
        ax2.set_ylim(0,60)
        ax2.set_ylabel('Temp. Méd Comp. Diária '+ '($^\circ$C)', fontsize=26)
        ax2.yaxis.set_tick_params(labelsize=23)
        
        counter+=1
        
            
    plt.tight_layout()
    plt.grid(True, axis = 'x')
    plt.grid(True, axis = 'y')
    
    filetarget = (os.path.join(root, "termo_plot.png"))
    plt.savefig(filetarget,dpi = 300)
    plt.show()

########################################### RUNNING CODE####################################

root = r'D:\estacoes_ba135'
root_mensal = r'D:\estacoes_ba135\mensais'
data,info_array = getclimate(root, False)

#key_to_output = 'Evaporacao Piche'
#info_single = get_series_queried_singlevalueinfo(data, info_array, key_to_output, startdate ='1988-1-1', enddate='2017-12-31', plot_original = False)

#key = 'Temp Comp Media'
#get_series_dadosmensais(data, key, attribute = 'mean', output = "info_mensal.csv",  killzero = False)

minimas = r'D:\Users\David Bispo\Desktop\minimas.csv'
maximas = r'D:\Users\David Bispo\Desktop\maximas.csv'
medias = r'D:\Users\David Bispo\Desktop\medias.csv'
evaporacao = r'D:\Users\David Bispo\Desktop\evaporacao.csv'
vento = r'D:\Users\David Bispo\Desktop\vento_max_med.csv'

#printa_serie_dadosmensais(arquivo = minimas, root = root, nplot = 2)

#processa_dfevap(data)
#printa_arquivoevap(evaporacao, output = "grafico_evaporacao.png")

#processavento(data)
#printavento(vento,root)

#processainsolacao(data)

#printa_termopluviometrico(data)
#printa_termopluvioevaporimetrico(data)