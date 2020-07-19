# Import dependencies.

import json
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import psycopg2
import time
from config2 import db_password

# Define a variable file_dir for the directory that’s holding our data.
file_dir = '/Users/chrissycho/Desktop/Movies-ETL'

#Open Json file and read into the variable file
with open(f'{file_dir}/wikipedia.movies.json', mode='r') as file:
    wiki_movies_raw = json.load(file)

# Extract, Transform and Load for 3 datasets
# ---EXTRACT----
# 1) Pull Wikipedia data into Pandas DataFrames
wiki_movies_df = pd.DataFrame(wiki_movies_raw)
# 2) Pull Kaggle data into Pandas DataFrames
kaggle_metadata = pd.read_csv(f'{file_dir}/movies_metadata.csv', low_memory=False)
ratings = pd.read_csv(f'{file_dir}/ratings.csv')

# ----TRANSFORM: Wikipedia---
# 1) Filter Wikipedia for movies using list comprehension
# Making a new list of movies with a movie link, director and removing TV shows
wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]

# Clean alternate titles nad merge the same information in different columns into one
def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    # combine alternate titles into one list
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune-Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles

    # merge column names
    def change_column_name(old_name, new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)
    change_column_name('Adaptation by', 'Writer(s)')
    change_column_name('Country of origin', 'Country')
    change_column_name('Directed by', 'Director')
    change_column_name('Distributed by', 'Distributor')
    change_column_name('Edited by', 'Editor(s)')
    change_column_name('Length', 'Running time')
    change_column_name('Original release', 'Release date')
    change_column_name('Music by', 'Composer(s)')
    change_column_name('Produced by', 'Producer(s)')
    change_column_name('Producer', 'Producer(s)')
    change_column_name('Productioncompanies ', 'Production company(s)')
    change_column_name('Productioncompany ', 'Production company(s)')
    change_column_name('Released', 'Release Date')
    change_column_name('Release Date', 'Release date')
    change_column_name('Screen story by', 'Writer(s)')
    change_column_name('Screenplay by', 'Writer(s)')
    change_column_name('Story by', 'Writer(s)')
    change_column_name('Theme music composer', 'Composer(s)')
    change_column_name('Written by', 'Writer(s)')

    return movie

# Appply the function to every movie in the dataframe using list comprehension
clean_movies = [clean_movie(movie) for movie in wiki_movies]
# Recreate the wiki_movies_df DataFrame with filtered information
wiki_movies_df = pd.DataFrame(clean_movies)

# Using regular expression Regex characters to print out imdb_link to a new column "imdb_id"
wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
print(len(wiki_movies_df))

# Drop any duplicates in the id numbers
wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
print(len(wiki_movies_df))

# Check any null columns
[[column,wiki_movies_df[column].isnull().sum()] for column in wiki_movies_df.columns]
# Keep the columns if the null values are less than 90% of the rows
wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
# Create a new Data Frame with the less # of null columns
wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]

# Convert types of columns
# 1) Check which columns need to be converted
wiki_movies_df.dtypes
# 2) Following columns must be converted
# Box office should be numeric.
# Budget should be numeric.
# Release date should be a date object.
# Running time should be numeric.

# 3) Convert and Parse "Box Office" column 
# 3-a) Drop any missing values
box_office = wiki_movies_df['Box office'].dropna() 
# 3-b) Make a function to return values that are not string
def is_not_a_string(x):
    return type(x) != str
box_office[box_office.map(is_not_a_string)]
# OR
box_office[box_office.map(lambda x: type(x) != str)]
# 3-c) There are lists in the values
#      Make a separator string and then call the join() method
box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)
# 3-d) Using regex to find characters that need to be replaced
# form_one contains values matching the form $123.4 million/billion
# form_two contains values matching the form $123,456,489
form_one = r'\$\d+\.?\d*\s*[mb]illion'
form_two= r'\$\d{1,3}(?:,\d{3})+'
# 3-e) Create Boolean Series and select the box office values that don't match either
matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)
box_office[~matches_form_one & ~matches_form_two]
# 3-f) Now we have a series that have values to be replaced
# Some values have spaces in between the dollar sign and the number. (\s*) * means 0 or more
# Some values have misspelling as "millon"
form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'
# Some values use a period as a thousands separator, not a comma.
form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'
# Some values are given a range
# search for any string that starts with a dollar sign and ends with a hyphen, 
# and then replace it with just a dollar sign
box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)

#3-g) Make a function that captures either form_one or form_two
def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.# million
    if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " billion"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a billion
        value = float(s) * 10**9

        # return value
        return value

    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

        # remove dollar sign and commas
        s = re.sub('\$|,','', s)

        # convert to float
        value = float(s)

        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan
# 3-h) Extract the values from box_office column using the function and drop the original "Box Office" column that's not parsed
wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
wiki_movies_df.drop('Box office', axis=1, inplace=True)

# 4) Parse the Budget Data
# 4-a) Create a budget variable and drop any missing values
budget = wiki_movies_df['Budget'].dropna()
# 4-b) Convert any lists to strings
budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)
# 4-c) Remove any values btw a dollar sign & a hyphen
budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)
# 4-d) Use the same pattern matches as box office data
matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
# 4-e) find paterns & remove the numbers in square brackets 
budget = budget.str.replace(r'\[\d+\]\s*', '')
# 4-f) select budget values that need to be replaced
budget[~matches_form_one & ~matches_form_two]
# 4-g) Extract the values from Budget column and change them to numeric values
wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)
# 4-h) Drop the original Budget column
wiki_movies_df.drop('Budget', axis=1, inplace=True)

# 5) Parse the Release Date
# 5-a) Make a variable that holds the non-null values of Release date
# in the DataFrae, converting lists to strings
release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
# 5-b) Parsing the following:
# Full month name, one- to two-digit day, four-digit year (i.e., January 1, 2000)
# Four-digit year, two-digit month, two-digit day, with any separator (i.e., 2000-01-01)
# Full month name, four-digit year (i.e., January 2000)
# Four-digit year
date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
date_form_two = r'\d{4}.[01]\d.[123]\d'
date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
date_form_four = r'\d{4}'
# 5-c) Pull the extracted info into a df
release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')
# 5-d) Parse the data using to_datetime() method in Pandas
wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)

# 6) Parse Running Time
# 6-a) Variable to hold the non-values of Running time in the DataFrame
# converting lists to strings
running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)
# 6-b) How many running times look like "100 minutes"
running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE).sum()
# 6-c) Look at what other 366 entries look like
running_time[running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE) != True]
# 6-d) more general: marking the beginning of the string, accepting other
# abbreviations of minutes by only searching up to the letter m
running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE).sum()
# 6-e) The ones don't match the regex (the remaining 17)
running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE) != True]
# 6-f) Match all of the regex
# Start with one digit.
# Have an optional space after the digit and before the letter “h.”
# Capture all the possible abbreviations of “hour(s).” To do this, we’ll make every letter in “hours” optional except the “h.”
# Have an optional space after the “hours” marker.
# Have an optional number of digits for minutes.
running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')
# 6-g) Convert to numeric values 
# Coercing the errors will turn the empty strings into NaN
# fillna() method to change all the NaNs to zeros
running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)
# 6-h) apply a function that will convert the hour capture groups 
# and minute capture groups to minutes if the pure minutes 
# capture group is zero
wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)
# 6-i) Drop the original column
wiki_movies_df.drop('Running time', axis=1, inplace=True)

# ----TRANSFORM: Kaggle Data---
# 1) Remove the bad data
kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]
# 2) Keep the movies with adult = False and drop the adult column
kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')
# 3-a) Clean the video column
# 3-b) Create a Boolean column for 'video' (assign it back to the same column)
kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'
# 4) Convert columns to numeric values
kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')
# 5) Convert release date to datetime
kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])

# ----TRANSFORM: Ratings Data---
# 1) Check rating data
ratings.info(null_counts=True)
# 2) Store the ratings data into its own SQL
# Convert timestamp to a datetime data type
pd.to_datetime(ratings['timestamp'], unit='s')
# 3) Assign the converted timestamp to the column
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

# ----TRANSFORM: Merge Wiki & Kaggle Data---
# 1) using inner join
movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])
# 2) Find redundnat columns
movies_df.columns
# Competing data:
# Wiki                     Movielens                Resolution
#--------------------------------------------------------------------------
# title_wiki               title_kaggle
# running_time             runtime
# budget_wiki              budget_kaggle
# box_office               revenue
# release_date_wiki        release_date_kaggle
# Language                 original_language
# Production company(s)    production_companies     
# 3) compare columns
# 3-a) Titles: print the wiki & kaggle titles data 
movies_df[['title_wiki','title_kaggle']]
# 3-a-1) rows that don't match
movies_df[movies_df['title_wiki'] != movies_df['title_kaggle']][['title_wiki','title_kaggle']] 
#second [['title_wiki','title_kaggle']] is needed to print out these two columsn only
# 3-a-2) Look for any missing data in the Kaggle titles
movies_df[(movies_df['title_kaggle'] == '') | (movies_df['title_kaggle'].isnull())]
# 3-b) Runtime: fill in missing values with zero and make the scatter plot
movies_df.fillna(0).plot(x='running_time', y='runtime', kind='scatter')
# 3-c) # Budget (numeric value)
movies_df.fillna(0).plot(x='budget_wiki',y='budget_kaggle', kind='scatter')
# 3-d-1) # Box Office
movies_df.fillna(0).plot(x='box_office', y='revenue', kind='scatter')
# 3-d-2) Box office anything less than 1 billion
movies_df.fillna(0)[movies_df['box_office'] < 10**9].plot(x='box_office', y='revenue', kind='scatter')
# 3-e-1) Release Date (in datetime not numeric so can't make a scatter plot but we can make a line graph with dots)
movies_df[['release_date_wiki','release_date_kaggle']].plot(x='release_date_wiki', y='release_date_kaggle', style='.')
# 3-e-2) Movies after 1996 for Wiki & before 1965 for Kaggle
movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')]
# 3-e-3) # The Holiday in the Wikipedia data got merged with From Here to Eternity
# Drop this row
movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index
movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)
# 3-e-4) Find null values
movies_df[movies_df['release_date_wiki'].isnull()]
movies_df[movies_df['release_date_kaggle'].isnull()]
# 3-f-1) Language
movies_df['Language'].value_counts()
# some are stored as lists 
# 3-f-2) Change the language data from lists to tuples
movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)
# 3-f-3) type of data in kaggle data -- no lists 
movies_df['original_language'].value_counts(dropna=False)
# 3-g-1) Production Companies
movies_df[['Production company(s)','production_companies']]

# ----TRANSFORM: Merge Wiki & Kaggle Data RESOLUTION---
# Resolutions (drop the columns)
movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)
# Function to fill in Wikipedia with zeros in Kaggle data
def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)

fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
movies_df
# check if any column has one value
# Convert list to tuple to run values_count()
for col in movies_df.columns:
    lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
    value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
    num_values = len(value_counts)
    if num_values == 1:
        print(col)
movies_df['video'].value_counts(dropna=False) # False for every row, so we don't need this column
# Re order columns in groups
# Identifying information (IDs, titles, URLs, etc.)
# Quantitative facts (runtime, budget, revenue, etc.)
# Qualitative facts (genres, languages, country, etc.)
# Business data (production companies, distributors, etc.)
# People (producers, director, cast, writers, etc.)
movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]]
# Rename to be consistent
movies_df.rename({'id':'kaggle_id',
                  'title_kaggle':'title',
                  'url':'wikipedia_url',
                  'budget_kaggle':'budget',
                  'release_date_kaggle':'release_date',
                  'Country':'country',
                  'Distributor':'distributor',
                  'Producer(s)':'producers',
                  'Director':'director',
                  'Starring':'starring',
                  'Cinematography':'cinematography',
                  'Editor(s)':'editors',
                  'Writer(s)':'writers',
                  'Composer(s)':'composers',
                  'Based on':'based_on'
                 }, axis='columns', inplace=True)

# ----TRANSFORM: Merge Rating Data 
# Indexing movie id with rating columns and counts for each rating value
rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count() \
                .rename({'userId':'count'}, axis=1) \
                .pivot(index='movieId',columns='rating', values='count')
# Rename columns using list comprehension
rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]

# ----TRANSFORM: Merge Movies_df (Kaggle & Wiki merged) with Rating data
# Left join
movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')
# Fill in 0s for missing values
movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)

# ----LOAD to PostgreSQL
# Make a connection string for the local server
connection_string = f'postgres://postgres:{db_password}@localhost:5433/movie_data'
# Create the database engine
engine = create_engine(connection_string)
# Import Movie Data
movies_df.to_sql(name='movies', con=engine)
# Import raw Ratings Data
import time #Usually with other dependencies

rows_imported = 0
# get the start_time from time.time()
start_time = time.time()
for data in pd.read_csv(f'{file_dir}/ratings.csv', chunksize=1000000):
    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
    data.to_sql(name='ratings', con=engine, if_exists='append')
    rows_imported += len(data)

    # add elapsed time to final print out
    print(f'Done. {time.time() - start_time} total seconds elapsed')