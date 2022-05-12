# Portfolio

This is a repository to highlight portfolio projects. 

<h3>Table of Contents:</h3>

1. [Project 1: Troll Tweet Analysis](#Project-1)
    * [Sample Visualizations](#Sample-1)
2. [Project 2: Absenteeism at Work Analysis](#Project-2)
   * [Sample Visualizations](#Sample-2)
3. [Project 3: Movie Media Analysis](#Project-3)
    * [Sample Visualizations](#Sample-3)
4. [Project 4: Microbe Analysis](#Project-4)
    * [Sample Visualizations](#Sample-4)

<a name="Project-1"></a>
<h2 align="center">Project 1: <a href="https://github.com/fahm1/Portfolio/tree/main/Troll%20Tweet%20Analysis">Troll Tweet Analysis</a></h2>
In this project, we will analyze Russian troll tweet data from the dataset provided by <a href="https://github.com/fivethirtyeight/russian-troll-tweets/">fivethirtyeight</a>.  
Some analysis of this data has already been <a href="http://pwarren.people.clemson.edu/Linvill_Warren_TrollFactory.pdf">completed and published</a> by Linvill and Warren of Clemson university.

</br>

In this analysis, we will be trying to answer one primary question:  
**What characteristics of troll accounts and their tweets make them successful?**

In this case, we are equating success to the number of followers the troll account has, as a higher follower account will allow the troll to spread their message to a greater number of individuals. 

As we answer this question, we will be answering some secondary questions that will help guide our answer, such as:
* Which account category is most successful?
* What are some specific characteristics of the most successful account category?
* What type of language are the top troll accounts using?

<a name="Sample-1"></a>
<h3 align="center">Sample Visualizations:</h3>
<h3 align="center">
<b>Sample 1:</b> Mixed Pairplot / KDE plot / Heatmap showing characteristics of Russian troll accounts and tweets.
</h3>
<p align="center"><img src="Troll%20Tweet%20Analysis/tweet_characteristics.png"/ width=100%></p>

<h3 align="center">
<b>Sample 2:</b> Boxplots showing the distribution of followers by account category.
</h3>
<p align="center"><img src="Troll%20Tweet%20Analysis/followers_by_category.png"/ width=100%></p>


<a name="Project-2"></a>
<h2 align="center">Project 2: <a href="https://github.com/fahm1/Portfolio/tree/main/Absenteeism%20Analysis">Absenteeism at Work Analysis</a></h2>
In this project, we will analyze the absenteeism data from the UCI Machine Learning Repository [found here](https://archive.ics.uci.edu/ml/datasets/Absenteeism+at+work).

The database was created with records of absenteeism at work from July 2007 to July 2010 at a courier company in Brazil.  

We will define absenteeism as follows:  
* absence from work during normal working hours resulting in temporary incapacity to execute regular working activity

In this analysis, we will be trying to answer one primary question:  
**Which factors are most indicative of excessive absenteeism from work and can we predict absenteeism by identifying these factors?**

While answering this question, we will also be making use of logistic regression in order to create a model which can predict absenteeism. 

By answering this question and creating our model, we will be able to predict the absenteeism rate for any given employee, which will help us maximize the quantity and quality of work done. 

Some visualizations shown will be created using Tableau.

[Click here to access an interactive Tableau dashboard](https://public.tableau.com/views/absentee_analysis_workbook/Dashboard?:language=en-US&:display_count=n&:origin=viz_share_link)

<a name="Sample-2"></a>
<h3 align="center">Sample Visualizations:</h3>
<h3 align="center">
<b>Sample 1:</b> Barplot showing the odds ratios of the various analyzed features in predicting absenteeism. 
</h3>
<p align="center"><img src="Absenteeism%20Analysis/feature_odds_ratios.png" width=100%></p>

<h3 align="center">
<b>Sample 2:</b> Primary features, age, and BMI vs absenteeism probability. 
</h3>
<p align="center"><img src="Absenteeism%20Analysis/dashboard.png"/ width=100%></p>


<a name="Project-3"></a>
<h2 align="center">Project 3: <a href="https://github.com/fahm1/Portfolio/tree/main/Movie%20Analysis">Movie Media Analysis</a></h2>
In this project, we will analyze movie data, specifically the film media (e.g. film vs. digital), the genre, and the budget.

In doing so, we will be trying to answer the following questions:
* How has the distribution of film media changed over time? 
* How has the distribution of genres changed over time?
* Does movie genre dictate film media?
* What are the distributions of budgets of each film media?

<a name="Sample-3"></a>
<h3 align="center">Sample Visualizations:</h3>
<h3 align="center">
<b>Sample 1:</b> Stacked bar chart of the distribution of movie genre over time from 2006-2017.
</h3>
<p align="center"><img src="Movie%20Analysis/Genre_by_Year.png"/ width=100%></p>

<h3 align="center">
<b>Sample 2:</b> Boxplots of the distribution of movie budgets by film media.
</h3>
<p align="center"><img src="Movie%20Analysis/Budget_by_Media.png"/ width=100%></p>


<a name="Project-4"></a>
<h2 align="center">Project 4: <a href="https://github.com/fahm1/Portfolio/tree/main/Microbe%20Analysis">Microbe Data Analysis</a></h2>
Data is adapted from the publicly available data for the <a href="https://www.pnas.org/content/107/14/6477">2010 Fierer et al. paper</a>.

In this project, we will analyze bacterial phyla that were found on keyboard keys as well as on finger tips. We will also take a look at this data on an individual basis. 

In doing so, we will be trying to answer the following questions:   
* Are some bacterial phyla highly correlated to another bacterial phyla?
* Do bacterial communities found on keyboard keys and fingertips differ on an individual basis?
* Is there a correlation between bacteria found on keyboard keys and on finger tips?

<a name="Sample-4"></a>
<h3 align="center">Sample Visualizations:</h3>
<h3 align="center">
<b>Sample 1:</b> Heatmap that shows the correlations between the relative abundances of various bacterial Phyla found on keyboard keys and fingertips.
</h3>
<p align="center"><img src="Microbe%20Analysis/heatmap_microbe.png"/ width=100%></p>

<h3 align="center">
<b>Sample 2:</b> Parallel coordinates plot to show how phylum abundances vary between individuals.
</h3>
<p align="center"><img src="Microbe%20Analysis/pcp_microbe.png"/></p>

<h3 align="center";>
<b>Sample 3:</b> 2D scatterplot showing bacterial grouping observations using PCA with colors to distinguish each of the three analyzed individuals.    
</h3>
<p align="center"><img src="Microbe%20Analysis/pca_indiv.png" width=800px></p>
