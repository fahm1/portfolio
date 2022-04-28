## Microbe Data Analysis

Data is adapted from the publicly available data for the [2010 Fierer et al. paper](https://www.pnas.org/content/107/14/6477)

In this project, we will analyze bacterial phyla that were found on keyboard keys as well as on finger tips. We will also take a look at this data on an individual basis. 

In doing so, we will be trying to answer the following questions:

* Are some bacteria highly correlated to another bacterial phyla?
* Do bacterial communities found on keyboard keys and fingertips differ on an individual basis?
* Is there a correlation between bacteria found on keyboard keys and on finger tips?
<p>&nbsp;</p>







<p align="center";>
<font size="5">Are some bacteria highly correlated to another bacterial phyla?</font>
</p>

<p align="center";>
<b>Plot 1:</b> Heatmap that shows the correlations between the relative abundances of the different Phyla
<img src="heatmap_microbe.png"/>
</p>

<p align="center">
We can see a significant positive correlation between the <i>Bacteroidetes</i> and <i>Firmucites</i> as well as a strong negative correlation between <i>Actinobacteria</i> and <i>Firmicutes</i>
</p>
<p>&nbsp;</p>






<p align="center";>
<font size="5">Do bacterial communities found on keyboard keys and fingertips differ on an individual basis?</font>
</p>

<p align="center";>
<b>Plot 2:</b> Parallel coordinates plot to show how phylum abundances vary between individuals
<img src="pcp_microbe.png"/>
</p>
<p align="center">
We can see see that individuals M3 and M9 are fairly similar, while individual M2 has a significantly different bacterial ecosystem
</p>
<p>&nbsp;</p>





<p align="center";>
<font size="5">Do bacterial communities found on keyboard keys and fingertips differ from one individual to another?</font>
</p>

<p align="center";>
<b>Plot 3:</b> 2D scatterplot showing all of the observations using PCA with colors to distinguish each of the three individuals
<img src="pca_indiv.png"/>
</p>

<p align="center">
We can see that there are 3 mostly distinct groups differing by individual. We can also see that the individual M2 is much more different than the other individuals, but that there is still some significant difference between individuals M3 and M9 as well. 
</p>
<p>&nbsp;</p>


<p align="center";>
<font size="5">Is there a correlation between bacteria found on keyboard keys and on finger tips?</font>
</p>

<p align="center";>
Now, instead of looking at the bacteria by individual, we want to see if there are differences between the bacteria found on keyboard keys and on finger tips.
</p>

<p align="center";>
<b>Plot 4:</b>2D scatterplot showing all of the observations using PCA with colors to distinguish sample locations
<img src="pca_loc.png"/>
</p>

<p align="center">
We can see that no distinct groups of bacteria were found on keyboard keys and on finger tips, implying that the bacteria found on keyboard keys and on finger tips are largely the same. 
</p>
<p>&nbsp;</p>

<p align="center";>
We can also take a look at using TSNE instead of PCA as a different form of dimensionality reduction to see if there is a difference in the bacteria found on keyboard keys and on finger tips.
</p>
<p align="center";>
In this case, we will look specifically at individual M3
</p>
<p>&nbsp;</p>

<p align="center";>
<b>Plot 4:</b>2D scatterplot showing all of the observations using TSNE with colors to distinguish sample locations
<img src="tsne_m3.png"/>
</p>

<p align="center">
Once again, no distinct groups are found, implying that the bacteria found on keyboard keys and on finger tips are likely one and the same. 
</p>
<p>&nbsp;</p>
