# Chrissy Cho's Movies-ETL
### Table of Contents
[ 1. Project Overview ](#desc)<br /> 
[ 2. Resources ](#resc)<br /> 
[ 3. Objectives ](#obj)<br /> 
[ 4. Summary ](#sum)<br /> 
[ 5. Challenge Overview ](#chal)<br /> 
[ 6. Challenge Summary ](#chalsum)<br /> 
[ 7. Challenge Assumption ](#assum)<br />

<a name="desc"></a>
## Project Overview
In this module, we've learned to extract the Wikipedia and Kaggle data from their respective files, transform the datasets by cleaning them up and joining them together, and load the cleaned dataset into a SQL database. This is a process to create data pipeline. This process allows cleaning any messay data into good data before performing data analysis. In this project, we are using movie data from Wikipedia and Kaggle for their metadata and rating data. 

<a name="resc"></a>
## Resources
- Data Source: [wikipedia.movies.json](https://github.com/chrissycho/Movies-ETL/blob/master/wikipedia.movies.json), [ratings.csv](https://github.com/chrissycho/Movies-ETL/blob/master/ratings.csv), [movies_metadata.csv](https://github.com/chrissycho/Movies-ETL/blob/master/ratings.csv)
- Software: [PostgreSQL version 12.3](https://www.enterprisedb.com/downloads/postgres-postgresql-downloads), [pgAdmin](https://www.postgresql.org/ftp/pgadmin/pgadmin4/v4.23/macos/), Python, Pandas

<a name="obj"></a>
## Objectives
- Create an automated ETL pipeline
- Extract data from multiple resources using Juypter Notebook
- Clean and transform the data automatically using Pandas and regular expressions
- Load new data into PostgreSQL

<a name="sum"></a>
## Summary
### ETL 
ETL is a process of extracting, transforming and loading data. 
![](pictures/ETL.png) 
![](pictures/Extract.png) 
During the Extract process, we will load data from mutiple sources into an interactive software (e.g., Jupyter Notebook). For this module, we extracted json data from Wikipedia and csv files from Kaggle. 

![](pictures/Transform.png)
![](pictures/Iterative.png)
During the Transform process, we've followed an iterative process of insepction, planning, and executing to clean data. Once we extract data, we many need to filter, parse, sort, pivot, summarize, or merge datasets to have a consistent and clean data. When we handle a data source, we many perform multiple iterative process of inspection, planning, and executing codes. Before making any changes to data, we must examine the data type, missing values, duplicates or etc. 

![](pictures/Load.png)
Once we identify problems, we will then plan out how to fix the data. We can make decisions such as dropping the entire column or fill in zeros with missing values. Then, we can write codes to carry out our plans. 

<a name="chal"></a>
## Challenge Overview
For this challenge, we will write a Python script that performs all three ETL steps on the Wikipedia and Kaggle data. We will leave out any code that performs exploratory data analysis, and we may need to add code to handle potentially unforeseen errors due to changes in the underlying data.

### Objectives
- Create an automated ETL pipeline.
- Extract data from multiple sources.
- Clean and transform the data automatically using Pandas and regular expressions.
- Load new data into PostgreSQL.

<a name="chalsum"></a>
## Challenge Summary
For the purpose of the challenge, we have performed etract, transformation and load similar to the way we did for the module. Refer to Challenge.ipynb and challenge.py for the codes and scripts. The codes have run without error. 

<a name="assum"></a>
## Challenge Assumptions
During the Merge process of Wiki's movie data and Kaggle's metadata, we had to make multiple decisions on how to resolve problems. We started by comparing columns that have similar data. For each pair of columns, we inspected data that is more consistent and has less outliers. It's best practice to document the data cleaning assumptions and decisions so that we can easily come back and change if necessary. 

For titles columns, we inspected both columns and based on the output, the Kaggle's title column had more consistent values compared to Wiki's data, especially Kaggle data had no missing values. Therefore, we've decided to drop the Wikipedia's title_wiki column. 

For the following columns, we inspected their scatter plot as it's a great way to inspect any outliers. The runtime, budget, and box office columns have shown that Kaggle data had less outliers but also had some missing values whereas those missing values have actual values in the Wiki data. We've decided to fill in those missing values with Wiki data. 

For the release date, we've performed a line plot to examine any outliers. There was one data that had two different movies mreged into one. We've decided to drop the row and keep the Kaggle data. Once we inspected the null values in both columns, the Wiki data had 11 missing values whereas Kaggle data had none. We've decided to drop the Wikipedia's release_date_wiki column.

For language columns, we've inspected that Kaggle column had the right data type whereas the Wiki data had mixture of lists. We've decided to drop the Wiki data.

For production columns, Kaggle's data had more consistent structure so we decided to drop Wiki data. 

In order to resolve the problems, we created a function to fill in any missing data with Wiki's data. 

def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)

For example, fill_missing_kaggle_data(movies_df, 'runtime', 'running_time') will return the wiki values if the kaggle value is equal to 0. 
