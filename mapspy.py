# -*- coding: utf-8 -*-
"""
Created on Wed March 12 20:05:13 2018

@author: Lenovo
"""
import pandas as pd
import folium
import matplotlib.pyplot as plt

data=pd.read_csv('Volcanoes_USA.txt')
latitude=list(data['LAT'])
longitude=list(data['LON'])
elev=list(data['ELEV'])
typ=list(data['TYPE'])
name=list(data['NAME'])
plt.hist(elev,bins=100)
plt.xlabel('Elevation in m')
plt.ylabel('freq')
plt.title('Volcanoes')
plt.show()




def color_producer(elev):
    if elev<1000:
        return 'green'
    elif 1000<=elev<3000:
        return 'orange'
    else:
        return 'red'
    

map=folium.Map(location=[19.045952, 72.889870],zoom_start=8,tiles='OpenStreetMap')
vesit=folium.FeatureGroup(name='Vesit')
vesit.add_child(folium.Marker(location=[19.045952, 72.889870],popup=folium.Popup('VESIT')))


taj=folium.FeatureGroup(name='Taj Mahal')
taj.add_child(folium.Marker(location=[27.1750199,78.0399665],popup=folium.Popup('The Taj Mahal')))
nike1=folium.FeatureGroup(name='NIKE STORE1')
nike1.add_child(folium.Marker(location=[19.0694347,72.8246892],popup=folium.Popup('Linking Road ,Apparel,SALE=50%off')))
nike2=folium.FeatureGroup(name='NIKE STORE2')
nike2.add_child(folium.Marker(location=[19.062073, 72.836191],popup=folium.Popup('Linking Road ,Bags,SALE=20%off')))
nike3=folium.FeatureGroup(name='NIKE STORE 3')
nike3.add_child(folium.Marker(location=[19.080691, 72.833479],popup=folium.Popup('Santacruz ,Shoes only,SALE=30%off')))

fgv=folium.FeatureGroup(name='Volcanoes')
for lat,longi,ele,ty in zip(latitude,longitude,elev,typ):
    fgv.add_child(folium.CircleMarker(location=[lat,longi],radius=6,popup=str(ty)+','+str(ele)+' m',
                                     fill_color=color_producer(ele),fill=True,color='black',fill_opacity=0.7))

fgp=folium.FeatureGroup(name='Population')
fgp.add_child(folium.GeoJson(data=open('world.json','r',encoding='utf-8-sig').read(),
                            style_function=lambda x:{'fillColor':'green'
                            if x['properties']['POP2005']>1000000000 else 'grey'  if 300000000<=x['properties']['POP2005']<1000000000 else 'red'
                            if 100000000<=x['properties']['POP2005']<300000000 else 'purple' if 70000000<=x['properties']['POP2005']<100000000 else 'yellow' if 30000000<= x['properties']['POP2005']<70000000
                            else 'blue' if 10000000<= x['properties']['POP2005']<30000000 else 'magenta' 
                            if  5000000<= x['properties']['POP2005']<10000000 else 'cyan' if 1000000<= x['properties']['POP2005']<5000000 
                            else 'orange'}))
    
map.add_child(vesit)   
map.add_child(fgv)
map.add_child(fgp)
map.add_child(taj)
map.add_child(nike1)
map.add_child(nike2)
map.add_child(nike3)
map.add_child(folium.LayerControl(position='topleft'))
map.save("Maps.html")