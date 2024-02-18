# Wild Forest Fire System Prediction using Neural Networks <br>
Clusterization <br>
Axel Daniela Campero Vega <br>
Department of Computer Science <br>
Golisano College of Computing and Information Sciences <br>
Rochester Institute of Technology <br>
Rochester, NY 14623, USA - 2023 <br>
ac6887@g.rit.edu <br>

## Introduction  
The development of artificial intelligence and Machine Learning has promoted the studies and data analysis to forecast the occurrence of forest fires. Machine learning algorithms are applied, modified or created to understand the complex interplay of multiple variables associated with wildfires [1]. The most used algorithms are neural networks, gradient boosting, k-nearest neighbors. However, no "best method" has been established for analyzing this problem. The works reviewed adopt specific development models, require high-capacity computing stations, use of specialized software, and extensive user coding experience.  <br>

This part of the project uses data from the FIRMS – NASA [2] satellite to carry out a geographical division into clusters of the Bolivian territory. The purpose is to establish the differences between the areas affected by forest fires to differentiate their frequency of occurrence, intensity and affected areas. To carry out this clustering, the K-Means [3] algorithm will be used. This valuable information will later be taken into account in the prediction and early warning system to mitigate the effect of forest fires in Bolivia.  <br>

## I. DATA ACQUISITION

NASA has implemented a system (FIRMS) [2], capable of monitoring forest fires globally and registering them in a freely accessible database throughout the world, which is an important source of data to study this problem in an attempt to find solutions that manage to mitigate it. Data acquisition is done by downloading FIRMS system files. The data has the format csv (comma-separated text files), and registers are also include Excel-type files format. FIRMS-Modis has data from 2000 to 2023 and they are arranged by country. <br>
<p align="center">
 <img width="452" alt="image" src="https://github.com/AxelCamperoVega/Clusterization-for-Wild-Fire-Forest-Prediction/assets/43591999/c276aff8-6ff5-4841-9ce5-a8cfc391adc0">

</p>
 
### Data Filtering
The following filters will be carried out in the pre-processing of the information: <br>
• Filtering of the attribute Type = 0 (Presumed vegetation fire)  <br>
• Filtering of the Confidence attribute (0-100%), which gives the probability that it is a forest fire. For security, values greater than or equal to 90% will be filtered, with which we can ensure that we include only the data on forest fires. <br>
• Geographical area filtering: This parameter does not have a specific indication in the FIRMS records. However, the latitude and longitude coordinates are available, which will be used to perform this filtering (clusterization). <br>
To filter the data a python program was developed [3]. The file filtered from 2022 has 76,714 records that we can consider forest fires clusterization in Bolivian territory with 90% certainty between the years indicated. To get an idea of its geographic distribution we plot the data. The data from the latitude and longitude columns of the fire historical data frame were used.  <br>

## II. CLUSTERIZACION
The term "cluster" is used in various disciplines to describe the grouping or classification of similar objects or data into larger sets. In the field of data science and data analytics, cluster refers to a group of objects or data points that share similar characteristics with each other and are different from objects or data points in other groups [4]. Clustering is an exploratory and descriptive technique, since it is not based on predefined labels or categories, but rather seeks to group data automatically and objectively. It is an unsupervised learning technique that aims to automatically identify and group similar objects or data points in the context of multiple sets. These clusters are formed based on data similarity or proximity, where data points within a cluster are more similar to each other than to data points in other clusters. Clustering allows us to discover patterns and structures of the data, allowing a deeper understanding of the information held in the large amount of data (in our case 76,714 samples of fires that occurred in Bolivia in 2022 with identification of the latitude and longitude). This technique is widely used in various areas, such as data mining, identification of areas with similar characteristics, bioinformatics, customer segmentation, anomaly detection and others.  <br>
Clustering algorithms seek to group data based on different criteria, such as Euclidean distance, cosine similarity or density measures. These algorithms can be partition-based, hierarchical-based, density-based, or model-based, where each has its own advantages and disadvantages in terms of scalability, sensitivity to initialization, and the shape of the clusters they can discover. Interpretation of the results depends on the context and specific objectives of the analysis. For this project, the identification of clusters will help to particularize the behavior of each cluster that presents differences in the number of fires, atmospheric pressure, relative humidity, anomalies and other factors that distinguish them from the rest of the clusters. <br> 
### CLUSTERIZATION ALGORITHMS
There are many Clustering algorithms, oriented to the type of clustering that is desired. Each algorithm has its own advantages, limitations and assumptions about the data to be worked with, in our case it is fire data produced and recorded in Bolivia with latitude and longitude information. The best known algorithms are:  <br>
**Hierarchical Clustering:** Builds a hierarchy of clusters in which objects are grouped at multiple levels of granularity. Works on the agglomerative algorithm and divisive algorithm.  <br>
**K-Means [4]:** It is one of the most popular clustering algorithms. Divides objects into k clusters, where k is an arbitrary predefined value. It seeks to minimize the sum of the squared distances between the objects and the centroids of the clusters. The algorithms it works with are K-means + + and K-medoids.  <br>
**DBSCAN (Density-Based Spatial Clustering of Applications with Noise):** This algorithm is based on the density of objects in the data space. Groups together objects that are close to each other and have a sufficiently high density. The Algorithms it works with are OPTICS (Ordering Points to Identify the Clustering Structure) and HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise).<br>
**Spectral Clustering:** Uses the similarity matrix of objects to group them. It considers the structure of the data and can handle non-linear or irregularly shaped clusters. The algorithms it works with are Normalized Cuts and Spectral Clustering.<br>
**Density-based clustering:** Groups objects into dense areas of the data space and separates them into less dense regions. The algorithms it works with are DBSCAN (mentioned above) and Mean-Shift.<br>
**Clustering by mixture models:** It is based on the idea that objects are distributed according to probability models. It seeks to fit a mixture model that represents the distribution of the data and assigns the objects to the different clusters. The algorithms it works with are Expectation-Maximization (EM) and Gaussian Mixture Models (GMM).<br>
<p align="center">
 <img width="355" alt="image" src="https://github.com/AxelCamperoVega/Clusterization-for-Wild-Fire-Forest-Prediction/assets/43591999/e5333c0e-dd61-450c-bea0-99d331e9995f">
 <br>Clustering Models

</p>
 

 


### DISTANCES AND SIMILARITIES
Clustering methods imply some measure of similarity between groups; the normal procedure is to use a distance measure. Distances play a critical role in determining similarity or dissimilarity between data points. There are several distance measures commonly used in clustering. The most used:  <br>
**Euclidean Distance:** It is the most used distance measure and is calculated as the square root of the sum of the squared differences between the coordinates of two points in an n-dimensional space. The formula for the Euclidean distance between two points, A(x1,y1,...,xn) y B(x2,y2,...,xn), is:  <br>
d(x,y)=√(∑_(i=1)^n▒(x_i-y_i )^2 )


### SELECTION OF THE CLUSTERING METHOD FOR THE STUDY AREA
Clearly the concentration of fire events has a function associated with its geographical position and this is because the geographical areas have a great variation in altitude (it can vary from 300m to 4,000m above sea level). Relative humidity, temperature and soil type are also very different, depending on the geographic region. However, the behavior of forest fires in their distribution form does not have anomalies and rather presents a simple distribution depending on the area considered. The data that we will use to carry out the clustering will be from the FIRMS-MODIS [2] satellite. These data have a probability of being true forest fires, rated as a percentage in NASA's own records, as seen in the data acquisition. For this grouping process, we will apply a filter to ensure that it is a forest fire, with more than 90% probability. It can also be observed that this phenomenon does not have a spectral, helical type behavior or an anomalous relationship between events that are very geographically separated. <br>
According to the above considerations, the best clustering option is K-Means [4], which can assign arbitrary the number of clusters K and minimize the sum of the squared distances between the objects and the centroids of the clusters. To define K, we can have as criteria their geographical extension and the homogeneity they present in their climatic conditions. Bolivia is politically divided into 9 departments. We will assume 10 as a reasonable number of clusters for this work. These clusters do not necessarily coincide with the political geographic division of Bolivia. Once the clusters are found, we will find the probability of fire occurrence for each one.<br>
### PROCEDURE
• A program is developed in python [3], capable of loading the original data in csv format, corresponding to one year of evaluation of forest fires in Bolivia (76,714 latitude and longitude samples). 
• The program implements K-Means with the help of Python libraries [3], identifies 10 clusters and provides 10 centroids (latitude and longitude), for each cluster.
• Once the centroids have been identified, an analysis of the area is carried out to associate each centroid with the closest site in order to define the cluster not only with a number, but with the area of influence of a specific city that facilitates management and fire prevention.<br>

### CLUSTERIZATION ALGORITHM AND PROGRAMMING
The algorithm begins by reading the data from the annual file of reports obtained from MODIS in “csv” format, loads them into a record and then assigns the value of K. The value of K was chosen as K=10, this value, despite be arbitrary, it may be optimal according to the elbow criterion. This is a technique that we use to determine the optimal number of centroids(k). To do this, we continuously iterate for k=1 to k=n, then the optimal value will be in the “elbow” of the graph. This verification is carried out to compare the choice of K=10, with the result of applying this technique. It can be seen in Figure 4 that the value of K=10 can be considered within the optimal values chosen for K.<br>
The program performs iterations to classify the data based on the minimum distance criterion, until finishing with the last one. Once the classification job is finished, the outputs are:<br>
• Diagram showing the “elbow” criterion <br>
List of the K centroids (latitude and longitude) <br>
• Map of the clustering of all data, superimposed on the geographical map of Bolivia. To achieve this, specialized python [3] libraries were used.<br>
• Excel table that indicates, for each original data, the classification that corresponds to it. This classification is presented as an association to the cluster number. However, as explained previously, prior work was carried out to associate the centroid found by K-Means to the nearest city to serve as a specific geographical reference. <br>

<p align="center">
<img width="274" alt="image" src="https://github.com/AxelCamperoVega/Clusterization-for-Wild-Fire-Forest-Prediction/assets/43591999/ee3edfec-3ae4-4b63-977a-4b2d64cecd32">

 <br> Flowchart for clustering program
</p>

PYTHON - Visual Studio Code is used and the necessary libraries are imported to work with numerical data, perform visualizations and manipulate data in table format.<br>
• NumPy: is a fundamental library for scientific computing in Python [3]. It provides efficient support for manipulating multidimensional arrays, along with an extensive collection of high-level mathematical functions for operating on these arrays.  <br>
• Matplotlib: is a 2D plotting library in Python that allows you to create high-quality visualizations. Matplotlib's pyplot sublibrary provides a MATLAB-like interface, making it easy to create graphs and diagrams. <br>
• Pandas: is a Python library used for data analysis and manipulation. It provides flexible and efficient data structures, such as DataFrames, that allow you to work with tabular data intuitively. Pandas offers a wide range of functionality for filtering, transforming, and manipulating data, making it an essential tool for data preparation and cleansing tasks in data analysis. <br>
• sklearn.cluster: is a module of the scikit-learn (also known as sklearn) library that provides clustering algorithms. Scikit-learn is a widely used library for machine learning in Python and offers a wide range of algorithms and tools for classification, regression, clustering tasks, and more. The sklearn.cluster module includes popular clustering algorithms such as k-means, DBSCAN, and agglomerative, which are used to identify patterns and clusters in unlabeled data sets. <br>
• k-Means, preprocessing and StandardScaler, are imported from sklearn.cluster to apply the selected clustering algorithm. <br>
• mpl_toolkits.basemap, is used to import the geographical map of Bolivia fully referenced in latitude and longitude and then overlay the clusters and fire data in the correct place, obtaining a more useful geographical reference. <br>
Special functions are also used to work with the database and csv format, following this procedure:
• The file download from MODIS and previously filtered is imported in .xlsx <br>
• The data_graf function creates a copy of the DataFrame data and assigns it to this variable. This copy will later be used for data visualization. <br>
The program goes ahead and runs the K-Means function, finds the 10 centroids and prints them to the screen. There are 10 clusters as established above. Output from program: <br>
[array([<br>
-20.26519467, -18.44224552, -17.6824001 , -16.89223428, -16.28382792, -15.73068698, -15.29447744, -14.02894278, -12.56058506, -12.33801936]), <br>
array([<br>
-63.72830764, -58.85526357, -61.48910434, -59.37772623, -63.84553085, -61.38659001, -66.90195189, -64.08237086,  -65.86662843, -68.05166652])]<br>
The algorithm parameter is set to "elkan", which is a more efficient algorithm for large data sets. The K-means [4] model is then fitted to the data, which means that it finds the clusters and assigns each data point to the corresponding cluster, assigning cluster labels to each data point. The variable tagged contains an array with the cluster labels for each point in the same order as the original data.<br>
We list and arrange in ascending order the towns/cities closest to the centroids. Subsequently, we obtain the 10 Centroids in vector form that represent latitude and longitude respectively and we customize the map graph with the functions of the Basemap library.<br>
• First, the x and y coordinates of the communities (or data points) are extracted and stored in the variables comn_x and comn_y, respectively. <br>
• A dictionary called data_to is created that contains information related to the clusters, in this way the original data (data_graf) is combined with the DataFrame data_to. <br>
• Using the pandas concat() function, the two DataFrames are concatenated horizontally (axis=1). <br>
• The ignore_index=True parameter ensures that indexes are reorganized correctly. <br>
• We print and save the .xlsx table <br>
The code saves the DataFrame data_out data to an Excel file. A pandas ExcelWriter object is created that will be used to write to an Excel file. The file is called "output.xlsx" and the writing engine 'openpyxl' is specified. Finally, the DataFrame data_out data is saved in an Excel spreadsheet within the file (exported from python program). <br>
<p align="center">
<img width="232" alt="image" src="https://github.com/AxelCamperoVega/Clusterization-for-Wild-Fire-Forest-Prediction/assets/43591999/a9ebfe9b-d7b4-4995-b1cb-66d9fd0bd22c"> <br> Sample of the output. The number of registers is 76,714.
</p>
 

Finally, the graph is obtained with the Centroids and the distributed data referring to each Cluster. <br> 

<p align="center">
<img width="196" alt="image" src="https://github.com/AxelCamperoVega/Clusterization-for-Wild-Fire-Forest-Prediction/assets/43591999/b298351d-0ec1-4f25-84ea-076f2ff46ff2">
<br> Clusterization, clusters and centroids/towns (output from Python program)
</p>
 

### ASSIGNMENT OF ZONES (CITIES) NEAR CENTROIDS
The proximity of a fire point to the centroid is determined using a distance measure, such as the Euclidean distance. Once the points are assigned to the initial centroids, the centroid of each group is calculated as the average of all the points assigned to that centroid. Then, the process of assigning and updating centroids is repeated until a convergence criterion is met. In this way, the population closest to the centroid is determined and grouped within the centroid that corresponds to it, assigning it the name of the Province and Department. The results are shown below and define the regions or clusters that will be taken into account for this work.  <br> 

<p align="center">
<img width="385" alt="image" src="https://github.com/AxelCamperoVega/Clusterization-for-Wild-Fire-Forest-Prediction/assets/43591999/25f66d82-db07-4b59-852c-b89f6777c37c">
<br> Centroids and sites
</p>


### RELATIVE FREQUENCY OF FIRE OCCURRENCE BY CLUSTER AND NORMALIZATION

The probability of having a fire event for each zone will be carried out by simple statistical counting, taking into account all records classified by regions or clusters. This gives us an idea of the density of events per cluster. It can be seen better in figure 5.7, where the relative frequencies per cluster are shown. We can see that regions have different relative probabilities of fire occurrence and this is because geographical, atmospheric and meteorological conditions differ over a wide range. <br>

<p align="center">
<img width="402" alt="image" src="https://github.com/AxelCamperoVega/Clusterization-for-Wild-Fire-Forest-Prediction/assets/43591999/59c9faad-575a-4c82-a19a-f2a9145175e7">

 <br> Fire in cluster probability Relative Frequency by Cluster
 </p>

It can be said that certain clusters will have a greater probability of forest fires than others. However, relative probabilities do not help much when you want to generalize or predict the probability of fire in a certain cluster, so these values will be scaled to classify them in a range from 0 to 1 (0% to 100%), using the maximum value obtained as a scale factor, which in this case is cluster 8, corresponding to the Ángel Sandoval region (zone). This represents an intermediate step in prediction since we can assign a normalized relative probability of fire occurrence based on the cluster, that is, its geographic location. This parameter will be used as input in the fire probability function along with other factors such as seasonal factor, atmospheric pressure and relative humidity. <br>


**References** <br>
<sup><sub>
[1] Predicting Forest Fires in Madagascar. Jessica Edwards, Manana Hakobyan, Alexander Lin and Christopher Golden. School of Engineering and Applied Sciences, Harvard University, Cambridge, MA, USA <br>
https://projects.iq.harvard.edu/files/cs288/files/madagascar_fires.pdf
[2] FIRMS (Fire Information Resource Management System – NASA)
      https://firms.modaps.eosdis.nasa.gov/country/ <br>
[3] KMeans Explained, Practical Guide To Data Clustering & How To In Python  https://spotintelligence.com/2023/08/23/kmeans/ <br>
[4] K-Means: MacQueen, J. B. (1967). Some Methods for classification and Analysis of Multivariate Observations. Proceedings of 5th Berkeley Symposium on Mathematical Statistics and Probability. Vol. 1. University of California Press. pp. 281–297. <br>
</sub></sup>




