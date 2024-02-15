# Wild Fire Early Warning - Clusterization
# Axel Daniela Campero 2023
# ROCHESTER INSTITUTE OF TECHNOLOGY
# Master in Computer Science Program

# Importing Python libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
import openpyxl
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

# We import the xlsx file with geographic coordinates
# of wild forest fire history to cluster (data from FIRMS - NASA satellite)

data = pd.read_excel(r'D:\Axel 2024\RIT Dic2022\Proyecto de grado\5 KMeans\DB.xlsx')
data_graf = data

# Data for graph
x = data['latitude'].values
y = data['longitude'].values

# Converting data from pandas to numpy, dfun = pd.DataFrame(data)

data = data.to_numpy()
data = list(data)

pob =list(np.zeros(76714))
dep = list(np.zeros(76714))

# Definitions
nc = range(1, 30) # Iterations we want to do
kmeans = [KMeans(n_clusters=i) for i in nc]
score = [kmeans[i].fit(data).score(data) for i in range(len(kmeans))]
score

# Applying KMeans Algorithm
kmeans = KMeans(n_clusters=10,algorithm="elkan")
kmeans.fit(data)
tagged = kmeans.labels_
centroids = kmeans.cluster_centers_

asc = centroids[centroids[:, 0].argsort()]
asc_x = asc[:,0]
asc_y = asc[:,1]
order = centroids[:, 0].argsort()
center = np.round_(centroids,decimals=2,out=None)  #center = np.sort(center)

print([asc_x, asc_y])

# Associating centroids to towns and states
for i in range(0,76714):
    if tagged[i]==order[0]:
        pob[i] = "Luis Calvo"
        dep[i] = "Chuquisaca"
    if tagged[i]==order[1]:
        pob[i] = "El Carmen rivero torrez"
        dep[i] = "Santa Cruz"
    if tagged[i]==order[2]:
        pob[i] = "Pailon"
        dep[i] = "Santa Cruz"
    if tagged[i]==order[3]:
        pob[i] = "Angel Sandoval"
        dep[i] = "Santa Cruz"
    if tagged[i]==order[4]:
        pob[i] = "San Pedro"
        dep[i] = "Santa Cruz"
    if tagged[i]==order[5]:
        pob[i] = "San Ignacio"
        dep[i] = "Santa Cruz"
    if tagged[i]==order[6]:
        pob[i] = "San Borja"
        dep[i] = "Beni"
    if tagged[i]==order[7]:
        pob[i] = "Huacareje"
        dep[i] = "Beni"
    if tagged[i]==order[8]:
        pob[i] = "Exaltación"
        dep[i] = "Beni"
    if tagged[i]==order[9]:
        pob[i] = "Ixiamas"
        dep[i] = "La Paz"

# Defining centroids to associate towns
cities = np.array([[-20.3172,-63.5823],
[-18.4371   ,-58.853],
[-17.703    ,-61.5823],
[-16.9105   ,-59.59420],
[-16.2691   ,-63.850],
[-15.7295   ,-61.388],
[-15.2990   ,-66.893],
[-14.0279   ,-64.081],
[-12.5605   ,-65.866],
[-12.3416   ,-68.051]])

# Storing tagged data
town_x = cities[:,0]
town_y = cities[:,1]

data_to = {'Cluster': tagged, 'Nearest town': pob, 'State': dep}
data_to = pd.DataFrame.from_dict(data_to)
data_out = pd.concat([data_graf,data_to],ignore_index=True,axis=1)
data_out.columns = ['Latitude', 'Longitude', 'No. Cluster', 'Nearest Town', 'State']


centroids_x = centroids[:,0]
centroids_y = centroids[:,1]
tag =   ["Luis Calvo","El Carmen", "Pailon", "Angel Sand.", "San Pedro",
"San Ignacio", "San Borja", "Huacaraje", "Exaltación", "Ixiamas"]

# Plotting map in base map
fig = plt.figure()
ax = fig.add_axes([0.1,0.1,0.8,0.8])
m = Basemap(projection='cyl', llcrnrlat=-23, urcrnrlat=-9,
llcrnrlon=-70.4, urcrnrlon=-56.8, resolution='f')

m.fillcontinents(color="#D5F5E3", lake_color="#3498DB")
m.drawmapboundary(fill_color="#17202A")
m.drawcoastlines()
m.drawcountries(color='#17202A')
m.drawstates(color="#17202A")

m.scatter(y, x,latlon=False, c=kmeans.labels_,s=7)
m.scatter(centroids_y, centroids_x,latlon=False,  marker = "s", s=20, 
    linewidths = 3, zorder = 20, c="darkkhaki")
m.scatter(town_y, town_x,latlon=False,  marker = "d", s=20,
    linewidths = 4, zorder = 10, c="turquoise")

for xi, yi, text in zip(town_y, town_x, tag):
    ax.annotate(text,
                xy=(xi, yi), xycoords='data',
                xytext=(0.5,-0.5), textcoords='offset points',zorder=20,fontname='Arial',bbox=dict(boxstyle="square,pad=0.15", alpha=0.25,fc="gray",
                ec="gray", lw=0))
for xi, yi, text in zip(asc_y, asc_x, order):
    ax.annotate(text, color = "red",fontsize='x-large',
                xy=(xi, yi), xycoords='data',
                xytext=(2,0), textcoords='offset points',zorder=24,fontname='Arial') 

plt.show()

# To export the data to excel
writer = pd.ExcelWriter("output.xlsx", engine='openpyxl')
data_out.to_excel(writer,sheet_name ='output', index=False)
writer.save()
#data_out.to_excel("output.xlsx",index=False)