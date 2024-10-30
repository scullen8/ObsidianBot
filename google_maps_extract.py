#%%
"""
I want to take all my points and create one large map showing all my webs of travel
maybe start with just plotting every point I have ever been
but then from there it would be cool to turn it into daily summaries/story of the travel

todo
save the parsed json data in a faster read load
save the json maps I like
ML to help me id flights
add end/start notation to the points
then figure out what was at the end/start points w google maps
correct date time for the timelines. Want to add the dt to the start t to figure out end time


"""

import json
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
from dateutil.parser import parse
from datetime import datetime
import math


def parse_point(coords):
    spl = coords.split(':')[1].split(',')
    return float(spl[0]), float(spl[1])

def parse_time(startTime, endTime):

    eT = parse(endTime)

    sT = parse(startTime)
    dT = eT-sT
    dT = dT.days*24.0+ dT.seconds/3600.0
    #print(f"Start: {sT}, End: {eT}, dT: {dT}")
    if dT<0 or math.isnan(dT):
        dT=0.0
    return sT, eT, dT

def parse_time2(startTime, endTime):

    eT = parse(endTime)

    sT = startTime
    dT = eT-sT
    dT = dT.days*24.0+ dT.seconds/3600.0
    #print(f"Start: {sT}, End: {eT}, dT: {dT}")
    if dT<0 or math.isnan(dT):
        dT=0.0
    return sT, eT, dT


def load_google_data(googleDataPath):
    """
    there seems to be 3 different data structure types as time went on
    """

    with open(googleDataPath,'r',) as f:
         mapData = json.load(f)

    total_points = {
        'lat':[],
        'lon':[],
        'start':[],
        'end':[],
        'dT':[]
        }

  
    for key in mapData:
        if 'visit' in key:
            x, y = parse_point(key['visit']['topCandidate']['placeLocation'])
            total_points['lat'].append(x)
            total_points['lon'].append(y)
            sT, eT, dT = parse_time(key['startTime'], key['endTime'])
            total_points['start'].append(sT)
            total_points['end'].append(eT)
            total_points['dT'].append(dT)

        elif 'activity' in key:
            x, y = parse_point(key['activity']['start'])
            total_points['lat'].append(x)
            total_points['lon'].append(y)
            total_points['end'][-1]
            sT, eT, dT = parse_time2(total_points['end'][-1], key['startTime'])
            total_points['start'].append(sT)
            total_points['end'].append(eT)
            total_points['dT'].append(dT)

            x, y = parse_point(key['activity']['end'])
            total_points['lat'].append(x)
            total_points['lon'].append(y)
            sT, eT, dT = parse_time(key['startTime'], key['endTime'])
            total_points['start'].append(sT)
            total_points['end'].append(eT)
            total_points['dT'].append(dT)



        elif 'timelinePath' in key:
            sT = parse(key['startTime'])
            eT = parse(key['endTime'])
            for kk in key['timelinePath']:
                x, y = parse_point(kk['point'])
                total_points['lat'].append(x)
                total_points['lon'].append(y)
                total_points['start'].append(sT)
                total_points['end'].append(eT)
                total_points['dT'].append(float(kk['durationMinutesOffsetFromStartTime'])/60.0)

    return total_points


def haversine_distance(metrics, showPlot=True):
    #convert out lat/lon in deg to rads
    metRads = np.radians(metrics)
    #calculate the sin^2 new lat/lon - old lat/lon /2
    metDiff = np.sin(np.diff(metRads, axis=1)/2)**2
    #print(metDiff)

    #calculate the cos lattitude terms
    metLatCos = np.cos(metRads[0,:])
    #print(metLatCos)
   
    #print(np.shape(metDiff))
    #add a zero to the front of metdiff so that our matrices line up
    metDiff = np.concatenate(([[0],[0]], metDiff), axis=1)
    #print(np.shape(metDiff))
    #print(np.shape(metLatCos))
    #make a copy of the cos lat values and shift down 1 so we can multiply across new lat old lat cos
    metLatCos2 = np.concatenate(([0], metLatCos[:-1]))
    #print(np.shape(metLatCos2))
    #finfin = np.prod(np.vstack([metLatCos, metLatCos2, metDiff[1,:]]), axis=0)
    #print("fin")
    #print(finfin)
    #print(np.shape(finfin))
    #finfin = np.prod(finfin, axis=0)
    #print(np.shape(finfin))
    #finfin = np.sum((metDiff[0,:], finfin), axis=0)
    #finfin = np.sum((metDiff[0,:], np.prod(np.vstack([metLatCos, metLatCos2, metDiff[1,:]]), axis=0)), axis=0)
    #print(np.shape(finfin))
    #havDist = 2* np.arcsin(np.sqrt(finfin))

    #combined Haversine formula for distance on Earth in Km
    havDistKilo = 12742* np.arcsin(np.sqrt(np.sum((metDiff[0,:], np.prod(np.vstack([metLatCos, metLatCos2, metDiff[1,:]]), axis=0)), axis=0)))
    
    if showPlot:
        logBins = [0.001, 0.01, 0.1, 1, 10, 100,1000, 10000]
        n, bins, patches = plt.hist(havDistKilo, bins=logBins, color='lightblue', ec='red')
        plt.xscale('log')
        for i in range(len(n)):
            print(f"Bin {logBins[i]}: {n[i]}")
        plt.show()
    return havDistKilo


def main():

    total_points = load_google_data("private/location-history.json")
    print(len(total_points['lat']))

    #metrics = np.array([total_points['lat'], total_points['lon']])
    #I want to subtract the last row from the next row and then take the sq dist
    #print(metrics)

    havDistKilo = haversine_distance(np.array([total_points['lat'], total_points['lon']]), False)

    print(havDistKilo)
    print(f"Median: {np.median(havDistKilo)}, Mean: {np.mean(havDistKilo)}\nMin: {np.min(havDistKilo)}, Max: {np.max(havDistKilo)}")

    speed = np.divide(np.array(havDistKilo), total_points['dT'], out=np.zeros_like(havDistKilo), where=np.array(total_points['dT'])!=0)

    print(speed)
    print(f"Median: {np.median(speed)}, Mean: {np.mean(speed)}\nMin: {np.min(speed)}, Max: {np.max(speed)}")
   # logBins = [0.001, 0.01, 0.1, 1, 10, 200,1000, 10000]
   # n, bins, patches = plt.hist(speed, bins=logBins, color='lightblue', ec='red')
   # plt.xscale('log')#print(f"Bin {logBins[i]}: {n[i]}")
   # plt.show()

    #print(total_points)

    print(np.shape(total_points['lat']))
    # gather the lat lon and distance to the next point. Also add a sequential counter so we can tell later if those points were connected
    fin_points = np.column_stack((np.arange(0, len(havDistKilo)), total_points['lat'], total_points['lon'], havDistKilo, speed))
    print(np.shape(fin_points))
    print(fin_points[:, 3])

    #km and km/hr thresholds for a data point to be a flight
    #we will need more complex filters to catch all the flight info, like group by day or decide based on points around it
    # this would be interesting for ML
    FLIGHT_THRESHOLD_SPEED = 200
    FLIGHT_THRESHOLD_DIST = 100
    flights = fin_points[np.where((fin_points[:, 4]>=FLIGHT_THRESHOLD_SPEED) | (fin_points[:, 3]>=FLIGHT_THRESHOLD_DIST))]
    grounds = fin_points[np.where((fin_points[:, 4]<FLIGHT_THRESHOLD_SPEED) & (fin_points[:, 3]<FLIGHT_THRESHOLD_DIST))]

    print(np.shape(grounds))
    #outP = np.where(fin_points[:, 2]>=200, )



    # https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_USA_0.json
    # https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_CAN_0.json

    
    geo_jsons = [ 
        'private/gadm41_USA_1.json',
        'private/gadm41_CAN_1.json'
        ]
    

   
    """
    geo_jsons = [
    'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_USA_1.json',
    'https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_CAN_1.json'
    ]
    """
    
    geo_df_total = pd.DataFrame()

    for geo_json in geo_jsons:
        geo_df = gpd.read_file(geo_json)
    
        geo_df_total = gpd.GeoDataFrame( pd.concat( [geo_df_total, geo_df], ignore_index=True) )

    print(geo_df_total.head())

    geo_df_total.plot()
    geo_axes = geo_df_total.plot()
    #geo_axes.scatter([-75], [40], c='red')
    #plt.show()
    #plt.savefig('map_with_single_marker_test.jpg')

    
    geo_axes = geo_df_total.plot(facecolor='#eaeaea', edgecolor='#fff', linewidth=.2, figsize=(5, 8), zorder=0)

    marker_size = 1

    #geo_axes.plot(total_points['lon'], total_points['lat'], c='teal', linewidth=.1, zorder=2)
    geo_axes.scatter(grounds[:, 2], grounds[:, 1], c='red', s=marker_size, marker='s', zorder=3, edgecolor='none')
    geo_axes.plot(flights[:, 2], flights[:, 1], c='teal', linewidth=.1, zorder=2)

    plt.gcf().axes[0].axis('off')
    plt.xlim(-125, -68)
    plt.ylim(25, 52)
    plt.show()
    
    



if __name__ == "__main__":
    main()