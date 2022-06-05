# Absenteeism as Work Analysis

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

---

<h2 align="center">Which factors are most indicative of excessive absenteeism from work and can we predict absenteeism by identifying these factors?</h2>
<h3 align="center">
<b>Plot 1:</b> Barplot showing the odds ratios of the various analyzed features in predicting absenteeism. 
</h3>

````{div} full-width
![](feature_odds_ratios.png)
````

<!-- <p align="center"><img src="feature_odds_ratios.png"/ width=100%></p> -->
We can see that 'Serious Symptoms', has an odds ratio of more than 22, suggesting that the model found that serious symptoms are the most important factor in determining absenteeism time, and as a result, it places the greatest weight with 'Serious Symptoms'.  
In fact, a person with serious symptoms is 22 times more likely to be absent than baseline. 
It is also worth noting that 'Various Diseases', is just behind serious symptoms and that there is a steep drop off in terms of importance after these two.  
With that, it's interesting to note that pregnancy and childbirth—which are typically thought to be a leading cause of absenteeism—are significant factor in absenteeism time, but not nearly as important as reasons 3 and 1. 

---

<h3 align="center">
<b>Plot 2:</b> Primary features, age, and BMI vs absenteeism probability. 
</h3>

````{div} full-width
![](dashboard.png)
````

<!-- <p align="center"><img src="dashboard.png"/ width=100%></p> -->
With the Primary Reasons vs Probability chart, we can analyze the four primary reasons for absenteeism and the probability of absenteeism for each of them. We can see that individuals with 'Serious Symptoms' or 'Various Diseases' have a very high likelihood of absenteeism, while individuals with 'Light Reasons' are far more likley to not show absenteeism.

With the Age & BMI vs Probability chart, we can actually see some interesting results, as we can see that the probability of absenteeism seems to remain just about constant for both increasing age and BMI. This may be a little surprising, as it might be expected that elderly individuals would have a higher likelihood of getting sick and therefore more absenteeism, but the data does not reflect this. The same could be said of BMI, where increasing BMI would be expected to result in greater absenteeism, but this is not observed. 