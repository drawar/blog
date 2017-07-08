---
layout: post
title:  "Predicting MLB Hall of Famers"
date:   2017-07-08
tags: spotify
image: /assets/article_images/2017-06-19-visualize-song-attributes-spotify/spotify-top-song.jpg
comments: true
---

## Introduction
Just because the election for 2017 crop of Hall of Famers only finished 5 months ago doesn't mean it's too early to start wondering which current major leaguers will be enshrined in Cooperstown someday. In fact it's only 5 months until the 2018 ballot is released so I think now is quite a felicitous time to examine previous ballots. Two questions I want to answer at the end of this post are:

1. Who got elected into the HoF and how similar/different were they from those who didn't?
2. Can we predict whose who will make it to the HoF?

Although there are four main position players can assume in a game of baseball: Infielders, Outfielders, Pitchers, and Catchers, the focus of this post will be on Infielders and Outfielders. This is one of the projects I did for *Udacity's Data Analyst Nanodegree*.

## Background information
The National Baseball Hall of Fame and Museum is located in Cooperstown, New York and was dedicated in 1939. A baseball player can be elected to the Hall of Fame if they meet the following criteria:

1. The player must have competed in at least ten seasons;
2. The player has been retired for at least five seasons;
3. A screening committee must approve the player’s worthiness to be included on the ballot and most players who played regularly for ten or more years are deemed worthy;
4. The player must not be on the ineligible list (that means that the player should not be banned from baseball);
5. A player is considered elected if he receives at least 75% of the vote in the election; and
6. A player stays on the ballot the following year if they receive at least 5% of the vote and can appear on ballots for a maximum of 10 years.

These criteria tell us what information we need to gather before answering our questions, namely how long each player competed, when they retired, whether they have been banned from the game, etc. In the next part, we are going to find out where to obtain these information.

## Dataset
The 2017 version of [Lahman's Baseball Database](http://www.seanlahman.com/baseball-archive/statistics/) contains complete batting and pitching statistics from 1871 to 2017, plus fielding statistics, standings, team stats, managerial records, post-season data, and more. The full database, as comma-separated files, can be downloaded from [here](http://seanlahman.com/files/database/baseballdatabank-2017.1.zip). However, for our predictions, we only need the following .csv files:
* Master.csv;
* Batting.csv;
* Fielding.csv;
* Teams.csv;
* AwardsPlayers.csv;
* AllstarFull.csv;
* Appearances.csv; and last but not least
* HallOfFame.csv.

Unfortunately, it's not possible to tell if a player has been banned from baseball from this database, but we can always look it up on the net. Next we will read in the data and clean them.

## Data Cleaning and Pre-processing


```python
# Import data to DataFrames
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read in the CSV files
master_df = pd.read_csv('Master.csv',usecols=['playerID','nameFirst','nameLast','bats','throws','finalGame'])
fielding_df = pd.read_csv('Fielding.csv',usecols=['playerID','A','E','DP'])
batting_df = pd.read_csv('Batting.csv', usecols = ['playerID', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI',\
                                                    'SB', 'CS', 'BB','SO', 'IBB', 'HBP', 'SH', 'SF'])
teams_df = pd.read_csv('Teams.csv')
awards_df = pd.read_csv('AwardsPlayers.csv', usecols=['playerID','awardID','yearID'])
allstar_df = pd.read_csv('AllstarFull.csv', usecols=['playerID','yearID'])
hof_df = pd.read_csv('HallOfFame.csv',usecols=['playerID','yearid','votedBy','needed_note','inducted','category'])
appearances_df = pd.read_csv('Appearances.csv')
```

In general, we are only interested in players elected by the BBWAA, but we should also include two players (Roberto Clemente and Lou Gehrig) who were elected via “Special Election”, because they clearly had Hall of Fame stats, but simply bypassed the process due to untimely circumstances.

Moreover, there were three occasions - in 1949, 1964, and 1967 - when the BBWAA conducted a special run-off election whereby the one player who received the most run-off votes would be elected to the HoF, so we should include players who got elected with a run-off ballot as well.


```python
hof = hof_df[((hof_df['votedBy'] == 'BBWAA') | (hof_df['votedBy'] == 'Special Election')) | (hof_df['votedBy'] == 'Run Off')]
hof = hof[(hof['category'] == 'Player') & (hof['inducted'] == 'Y')]

# Drop these columns as no longer useful
hof = hof.drop(['category', 'inducted', 'needed_note', 'yearid'], axis = 1)

# Convert `votedBy` column to numeric
hof['votedBy'] = 1

# Give `votedBy` column a better name
hof.rename(columns = {'votedBy':'HoF'}, inplace = True)
```

Next we'll gather information about each player's performance, starting with batting statistics:


```python
# Group by playerID
batting = batting_df.groupby('playerID', as_index = False).sum()
batting = batting.fillna(0)
batting.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>AB</th>
      <th>R</th>
      <th>H</th>
      <th>2B</th>
      <th>3B</th>
      <th>HR</th>
      <th>RBI</th>
      <th>SB</th>
      <th>CS</th>
      <th>BB</th>
      <th>SO</th>
      <th>IBB</th>
      <th>HBP</th>
      <th>SH</th>
      <th>SF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>aardsda01</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>aaronha01</td>
      <td>12364</td>
      <td>2174</td>
      <td>3771</td>
      <td>624</td>
      <td>98</td>
      <td>755</td>
      <td>2297.0</td>
      <td>240.0</td>
      <td>73.0</td>
      <td>1402</td>
      <td>1383.0</td>
      <td>293.0</td>
      <td>32.0</td>
      <td>21.0</td>
      <td>121.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>aaronto01</td>
      <td>944</td>
      <td>102</td>
      <td>216</td>
      <td>42</td>
      <td>6</td>
      <td>13</td>
      <td>94.0</td>
      <td>9.0</td>
      <td>8.0</td>
      <td>86</td>
      <td>145.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>aasedo01</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>abadan01</td>
      <td>21</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>4</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



The columns have somewhat cryptic names, but you can always take a look at the information at the [README page](http://seanlahman.com/files/database/readme2016.txt) to see what they mean. 

Next up is fielding statistics:


```python
# Group by playerID
fielding = fielding_df.groupby('playerID', as_index = False).sum()
fielding = fielding.fillna(0)
fielding.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>A</th>
      <th>E</th>
      <th>DP</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>aardsda01</td>
      <td>29.0</td>
      <td>3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>aaronha01</td>
      <td>429.0</td>
      <td>144.0</td>
      <td>218.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>aaronto01</td>
      <td>113.0</td>
      <td>22.0</td>
      <td>124.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>aasedo01</td>
      <td>135.0</td>
      <td>13.0</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>abadan01</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



Next, we will look at the `allstar_df` DataFrame. It contains information on which players made appearances in Allstar games. The Allstar game is an exhibition game played each year at mid-season. Major League Baseball consists of two leagues: the American league and the National league. The top 25 players from each league are selected to represent their league in the Allstar game. Hence to make appearances in the Allstar game is quite an achievement and we want to know how many Allstar games each player has participated in. 


```python
allstar = allstar_df.groupby('playerID').count().reset_index()
```

It might be a good idea to rename the column 'yearID' to something else to avoid confusion.


```python
allstar.rename(columns = {'yearID':'years_allstar'}, inplace = True)
```

Next up is `awards_df`, let's see how many different awards there are


```python
awards_df['awardID'].unique()
```




    array(['Pitching Triple Crown', 'Triple Crown',
           'Baseball Magazine All-Star', 'Most Valuable Player',
           'TSN All-Star', 'TSN Guide MVP',
           'TSN Major League Player of the Year', 'TSN Pitcher of the Year',
           'TSN Player of the Year', 'Rookie of the Year', 'Babe Ruth Award',
           'Lou Gehrig Memorial Award', 'World Series MVP', 'Cy Young Award',
           'Gold Glove', 'TSN Fireman of the Year', 'All-Star Game MVP',
           'Hutch Award', 'Roberto Clemente Award', 'Rolaids Relief Man Award',
           'NLCS MVP', 'ALCS MVP', 'Silver Slugger', 'Branch Rickey Award',
           'Hank Aaron Award', 'TSN Reliever of the Year',
           'Comeback Player of the Year', 'Outstanding DH Award',
           'Reliever of the Year Award'], dtype=object)



That's a *lot* of awards, but not all of them are correlated with being voted into HoF. In fact, let's just focus on the more important ones, namely: Most Valuable Player, Rookie of the Year, Gold Glove, Silver Slugger, and World Series MVP awards. (Cy Young, though being a major award, is only for pitchers and thus excluded.) Now we need to count how many different awards each player managed to win.


```python
# Keeping only important awards
awards_list = ['Most Valuable Player','Rookie of the Year','Gold Glove','Silver Slugger','World Series MVP']
awards = awards_df[awards_df['awardID'].isin(awards_list)]

# Pivot the data frame to count the number of different awards
awards = awards.pivot_table(index = 'playerID', columns = 'awardID', aggfunc='count')

# Flatten the pivot table
awards = pd.DataFrame(awards.to_records())
```

Notice that we have inadvertently introduced a decent number of NA values that are actually zeros when making the pivot table, so we'll have to replace them accordingly. We have also changed the column names as a result of flattening our pivot table. The simplest way to fix this is by string match-and-replace. 


```python
# Fix column names after flattening
awards.columns = [col.replace("('yearID', '", "").replace("')", "") \
                     for col in awards.columns]

awards = awards.fillna(0)
```

At this point we have gathered quite a decent amount of information on players' statistics, it's a good idea to try and compile them together:


```python
player_stats = batting.merge(fielding, on = 'playerID', how ='left')
player_stats = player_stats.merge(allstar, on = 'playerID', how ='left')
player_stats = player_stats.merge(awards, on = 'playerID', how ='left')
player_stats = player_stats.merge(hof, on = 'playerID', how ='left')
player_stats = player_stats.fillna(0)
player_stats.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>AB</th>
      <th>R</th>
      <th>H</th>
      <th>2B</th>
      <th>3B</th>
      <th>HR</th>
      <th>RBI</th>
      <th>SB</th>
      <th>CS</th>
      <th>...</th>
      <th>A</th>
      <th>E</th>
      <th>DP</th>
      <th>years_allstar</th>
      <th>Gold Glove</th>
      <th>Most Valuable Player</th>
      <th>Rookie of the Year</th>
      <th>Silver Slugger</th>
      <th>World Series MVP</th>
      <th>HoF</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>aardsda01</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>29.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>aaronha01</td>
      <td>12364</td>
      <td>2174</td>
      <td>3771</td>
      <td>624</td>
      <td>98</td>
      <td>755</td>
      <td>2297.0</td>
      <td>240.0</td>
      <td>73.0</td>
      <td>...</td>
      <td>429.0</td>
      <td>144.0</td>
      <td>218.0</td>
      <td>25.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>aaronto01</td>
      <td>944</td>
      <td>102</td>
      <td>216</td>
      <td>42</td>
      <td>6</td>
      <td>13</td>
      <td>94.0</td>
      <td>9.0</td>
      <td>8.0</td>
      <td>...</td>
      <td>113.0</td>
      <td>22.0</td>
      <td>124.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>aasedo01</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>135.0</td>
      <td>13.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>abadan01</td>
      <td>21</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



We also need to know when a player played their last game, these can be found in the `master_df` DataFrame:


```python
master_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>nameFirst</th>
      <th>nameLast</th>
      <th>bats</th>
      <th>throws</th>
      <th>finalGame</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>aardsda01</td>
      <td>David</td>
      <td>Aardsma</td>
      <td>R</td>
      <td>R</td>
      <td>2015-08-23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>aaronha01</td>
      <td>Hank</td>
      <td>Aaron</td>
      <td>R</td>
      <td>R</td>
      <td>1976-10-03</td>
    </tr>
    <tr>
      <th>2</th>
      <td>aaronto01</td>
      <td>Tommie</td>
      <td>Aaron</td>
      <td>R</td>
      <td>R</td>
      <td>1971-09-26</td>
    </tr>
    <tr>
      <th>3</th>
      <td>aasedo01</td>
      <td>Don</td>
      <td>Aase</td>
      <td>R</td>
      <td>R</td>
      <td>1990-10-03</td>
    </tr>
    <tr>
      <th>4</th>
      <td>abadan01</td>
      <td>Andy</td>
      <td>Abad</td>
      <td>L</td>
      <td>L</td>
      <td>2006-04-13</td>
    </tr>
  </tbody>
</table>
</div>



Data in `bats` and `throws` columns are binary values with `R` (`L`) indicating a player's batting/throwing hand is his right (left), so it's much simpler to represent the information with 0-1 integers.


```python
# Create a function to convert the `bats` and `throws` colums to numeric
def bats_throws(col):
    if col == "R":
        return 1
    else:
        return 0
    
#player_stats['bats_R'] = player_stats['bats'].apply(bats_throws)
#player_stats['throws_R'] = player_stats['throws'].apply(bats_throws)

master_df['bats_R'] = master_df['bats'].apply(bats_throws)
master_df['throws_R'] = master_df['throws'].apply(bats_throws)

# Drop the old columns
master_df = master_df.drop(['bats','throws'], axis = 1)
```

Moreover, the `debut` and `finalGame` columns are currently strings so we'll need to convert them to datetime object and extract the year, since we don't need details as granular as date and month.


```python
from datetime import datetime

def getYear(datestring):
    return datetime.strptime(datestring, '%Y-%m-%d').year

# Drop rows that have NA values
master = master_df.dropna(subset = ['finalGame'])

# Get years from strings
master = master.join(master['finalGame'].map(getYear), lsuffix='_')
master = master.drop('finalGame_', axis = 1)
```


```python
master.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>nameFirst</th>
      <th>nameLast</th>
      <th>bats_R</th>
      <th>throws_R</th>
      <th>finalGame</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>aardsda01</td>
      <td>David</td>
      <td>Aardsma</td>
      <td>1</td>
      <td>1</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>1</th>
      <td>aaronha01</td>
      <td>Hank</td>
      <td>Aaron</td>
      <td>1</td>
      <td>1</td>
      <td>1976</td>
    </tr>
    <tr>
      <th>2</th>
      <td>aaronto01</td>
      <td>Tommie</td>
      <td>Aaron</td>
      <td>1</td>
      <td>1</td>
      <td>1971</td>
    </tr>
    <tr>
      <th>3</th>
      <td>aasedo01</td>
      <td>Don</td>
      <td>Aase</td>
      <td>1</td>
      <td>1</td>
      <td>1990</td>
    </tr>
    <tr>
      <th>4</th>
      <td>abadan01</td>
      <td>Andy</td>
      <td>Abad</td>
      <td>0</td>
      <td>0</td>
      <td>2006</td>
    </tr>
  </tbody>
</table>
</div>



Next up is the `appearances_df` DataFrame. This contains information on how many appearances each player had at each position for each year and will tell us how long a player has competed in the game. Let's take a look at the first few rows of the DataFrame to see what we have.


```python
appearances_df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>yearID</th>
      <th>teamID</th>
      <th>lgID</th>
      <th>playerID</th>
      <th>G_all</th>
      <th>GS</th>
      <th>G_batting</th>
      <th>G_defense</th>
      <th>G_p</th>
      <th>G_c</th>
      <th>...</th>
      <th>G_2b</th>
      <th>G_3b</th>
      <th>G_ss</th>
      <th>G_lf</th>
      <th>G_cf</th>
      <th>G_rf</th>
      <th>G_of</th>
      <th>G_dh</th>
      <th>G_ph</th>
      <th>G_pr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1871</td>
      <td>TRO</td>
      <td>NaN</td>
      <td>abercda01</td>
      <td>1</td>
      <td>NaN</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1871</td>
      <td>RC1</td>
      <td>NaN</td>
      <td>addybo01</td>
      <td>25</td>
      <td>NaN</td>
      <td>25</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>22</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1871</td>
      <td>CL1</td>
      <td>NaN</td>
      <td>allisar01</td>
      <td>29</td>
      <td>NaN</td>
      <td>29</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29</td>
      <td>0</td>
      <td>29</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1871</td>
      <td>WS3</td>
      <td>NaN</td>
      <td>allisdo01</td>
      <td>27</td>
      <td>NaN</td>
      <td>27</td>
      <td>27</td>
      <td>0</td>
      <td>27</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1871</td>
      <td>RC1</td>
      <td>NaN</td>
      <td>ansonca01</td>
      <td>25</td>
      <td>NaN</td>
      <td>25</td>
      <td>25</td>
      <td>0</td>
      <td>5</td>
      <td>...</td>
      <td>2</td>
      <td>20</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# Drop unnecessary columns
appearances_df = appearances_df.drop(['G_ph', 'G_pr'], axis = 1)
appearances = appearances_df.groupby(['playerID'], as_index = False).sum()
appearances = appearances.fillna(0)
```


```python
appearances_df.columns
```




    Index(['yearID', 'teamID', 'lgID', 'playerID', 'G_all', 'GS', 'G_batting',
           'G_defense', 'G_p', 'G_c', 'G_1b', 'G_2b', 'G_3b', 'G_ss', 'G_lf',
           'G_cf', 'G_rf', 'G_of', 'G_dh'],
          dtype='object')



As mentioned earlier, this post is focused on infielders and outfielders only, so we need to pick players who only play at these positions. However, some players played at multiple different positions in the earlier years of MLB, so how are we going to filter out pitchers and catchers? There are no hard and fast rules, but we can convert these numbers into percentages and exclude people who played more than, say, 10% of their games at either of these positions.


```python
positions = appearances_df.columns[5:]

# Loop through the list and divide each column by the players total games played
for col in positions:
    column = col + '_percent'
    appearances[column] = appearances[col] / appearances['G_all'] 
```


```python
# Eliminate players who played 10% or more of their games as Pitchers or Catchers
appearances = appearances[(appearances['G_p_percent'] < 0.1) & (appearances['G_c_percent'] < 0.1)]

# Drop columns that are no longer useful
appearances = appearances.drop(['G_p_percent','G_c_percent', 'yearID'], axis = 1)
appearances = appearances.drop(positions.tolist(), axis = 1)
```


```python
appearances.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>G_all</th>
      <th>GS_percent</th>
      <th>G_batting_percent</th>
      <th>G_defense_percent</th>
      <th>G_1b_percent</th>
      <th>G_2b_percent</th>
      <th>G_3b_percent</th>
      <th>G_ss_percent</th>
      <th>G_lf_percent</th>
      <th>G_cf_percent</th>
      <th>G_rf_percent</th>
      <th>G_of_percent</th>
      <th>G_dh_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>aaronha01</td>
      <td>3298</td>
      <td>0.962098</td>
      <td>1.0</td>
      <td>0.905094</td>
      <td>0.063675</td>
      <td>0.013038</td>
      <td>0.002122</td>
      <td>0.000000</td>
      <td>0.095512</td>
      <td>0.093390</td>
      <td>0.659187</td>
      <td>0.836871</td>
      <td>0.060946</td>
    </tr>
    <tr>
      <th>2</th>
      <td>aaronto01</td>
      <td>437</td>
      <td>0.471396</td>
      <td>1.0</td>
      <td>0.791762</td>
      <td>0.530892</td>
      <td>0.016018</td>
      <td>0.022883</td>
      <td>0.000000</td>
      <td>0.308924</td>
      <td>0.002288</td>
      <td>0.004577</td>
      <td>0.313501</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>abadan01</td>
      <td>15</td>
      <td>0.266667</td>
      <td>1.0</td>
      <td>0.600000</td>
      <td>0.533333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.066667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>abadijo01</td>
      <td>12</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>abbated01</td>
      <td>855</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.490058</td>
      <td>0.023392</td>
      <td>0.453801</td>
      <td>0.000000</td>
      <td>0.002339</td>
      <td>0.001170</td>
      <td>0.003509</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



The data look fine but... how are we going to deal with the years? It is unlikely that the number 1950 will have the same relationship to the rest of the data that the model will infer, so is it OK to drop them like we just did? 

No it's not. It turns out as MLB progressed, different eras emerged where the amount of runs per game increased or decreased significantly. This means that when a player played has a large influence on that player’s career statistics. The HoF voters take this into account when voting players in, so our model needs that information too. To get information such as runs allowed and games played over the years, we need to turn to the `teams_df` DataFrame.

We only need to consider columns needed to calculate runs per game per year, the rest we can safely ignore. Also looking back at the history of MLB, the rules of baseball had not settled into place before 1900 and the game was a totally different beast back then so it makes sense to remove these rows from the data.


```python
# Runs and games per year
runs_games = teams_df.groupby('yearID').sum()[['G','R']]
runs_games['RPG'] = runs_games['R'] / runs_games['G']
runs_games = runs_games.loc[1901:]
runs_games.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>G</th>
      <th>R</th>
      <th>RPG</th>
    </tr>
    <tr>
      <th>yearID</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1901</th>
      <td>2220</td>
      <td>11068</td>
      <td>4.985586</td>
    </tr>
    <tr>
      <th>1902</th>
      <td>2230</td>
      <td>9883</td>
      <td>4.431839</td>
    </tr>
    <tr>
      <th>1903</th>
      <td>2228</td>
      <td>9892</td>
      <td>4.439856</td>
    </tr>
    <tr>
      <th>1904</th>
      <td>2498</td>
      <td>9307</td>
      <td>3.725781</td>
    </tr>
    <tr>
      <th>1905</th>
      <td>2474</td>
      <td>9640</td>
      <td>3.896524</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot number of runs per game over time
runs_games['RPG'].plot()
plt.title('MLB Yearly Runs per Game')
plt.xlabel('Year')
plt.ylabel('MLB Runs per Game')
plt.axvspan(1901, 1920, color='red', alpha=0.4)
plt.axvspan(1942, 1945, color='red', alpha=0.4)
plt.axvspan(1963, 1976, color='red', alpha=0.4)
plt.axvspan(1993, 2009, color='red', alpha=0.4)
```




    <matplotlib.patches.Polygon at 0x178c12390>




![png]({{ site.url }}/assets/article_images/2018-07-08-predict-mlb-hof/p2-investigate-data-set_39_1.png)


There were indeed some periods when the number of runs per game was much higher than others. For example, the years from 1920 - 1941 saw an unprecedented high number of runs scored per game and was often referred to as the Lively Ball Era. Another sharp rise in runs per game occurred during early '90s to 2008, the Steroid Era. To capture this information, we need to convert years to eras in our player_stats DataFrame and turn them into new features (columns). We can re-use part of the codes we wrote for awards_df to accomplish this.


```python
yr_appearances = appearances_df.copy()[['yearID','playerID','teamID']]

# Remove players in or before 1900
yr_appearances = yr_appearances[yr_appearances['yearID'] > 1900]

def toEra(year):
    if int(year) < 1921:
        return '1901-1920'
    elif int(year) < 1942:
        return '1921-1941'
    elif int(year) < 1946:
        return '1942-1945'
    elif int(year) < 1963:
        return '1946-1962'
    elif int(year) < 1977:
        return '1963-1976'
    elif int(year) < 1993:
        return '1977-1992'
    elif int(year) < 2010:
        return '1993-2009'
    else:
        return 'post2009'

yr_appearances['yearID'] = yr_appearances['yearID'].map(toEra)

# Pivot the data frame to count the number of different awards
yr_appearances = yr_appearances.pivot_table(index='playerID', columns = 'yearID', aggfunc='count')

# Flatten the pivot table
yr_appearances = pd.DataFrame(yr_appearances.to_records())

# Fix column names after flattening
yr_appearances.columns = [col.replace("('teamID', '", "").replace("')", "") \
                     for col in yr_appearances.columns]

yr_appearances = yr_appearances.fillna(0)

# Number of years playing
yr_appearances['years_playing'] = yr_appearances.sum(axis = 1)

yr_appearances = yr_appearances.merge(appearances, on = 'playerID', how = 'inner')
yr_appearances.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>1901-1920</th>
      <th>1921-1941</th>
      <th>1942-1945</th>
      <th>1946-1962</th>
      <th>1963-1976</th>
      <th>1977-1992</th>
      <th>1993-2009</th>
      <th>post2009</th>
      <th>years_playing</th>
      <th>...</th>
      <th>G_defense_percent</th>
      <th>G_1b_percent</th>
      <th>G_2b_percent</th>
      <th>G_3b_percent</th>
      <th>G_ss_percent</th>
      <th>G_lf_percent</th>
      <th>G_cf_percent</th>
      <th>G_rf_percent</th>
      <th>G_of_percent</th>
      <th>G_dh_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>aaronha01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>14.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>23.0</td>
      <td>...</td>
      <td>0.905094</td>
      <td>0.063675</td>
      <td>0.013038</td>
      <td>0.002122</td>
      <td>0.000000</td>
      <td>0.095512</td>
      <td>0.093390</td>
      <td>0.659187</td>
      <td>0.836871</td>
      <td>0.060946</td>
    </tr>
    <tr>
      <th>1</th>
      <td>aaronto01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.0</td>
      <td>...</td>
      <td>0.791762</td>
      <td>0.530892</td>
      <td>0.016018</td>
      <td>0.022883</td>
      <td>0.000000</td>
      <td>0.308924</td>
      <td>0.002288</td>
      <td>0.004577</td>
      <td>0.313501</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>abadan01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>0.600000</td>
      <td>0.533333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.066667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>abbated01</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.490058</td>
      <td>0.023392</td>
      <td>0.453801</td>
      <td>0.000000</td>
      <td>0.002339</td>
      <td>0.001170</td>
      <td>0.003509</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>abbotje01</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>...</td>
      <td>0.793991</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.270386</td>
      <td>0.347639</td>
      <td>0.236052</td>
      <td>0.793991</td>
      <td>0.051502</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



Now that we have gathered pretty much all necessary information, it's time for a final merge. It's likely that new NA values will be created as a result of merging, so we need to check if there are any of them as well.


```python
df = master.merge(player_stats, on = 'playerID', how ='left')
df = df.merge(yr_appearances, on = 'playerID', how ='inner')
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 7237 entries, 0 to 7236
    Data columns (total 53 columns):
    playerID                7237 non-null object
    nameFirst               7235 non-null object
    nameLast                7237 non-null object
    bats_R                  7237 non-null int64
    throws_R                7237 non-null int64
    finalGame               7237 non-null int64
    AB                      7237 non-null float64
    R                       7237 non-null float64
    H                       7237 non-null float64
    2B                      7237 non-null float64
    3B                      7237 non-null float64
    HR                      7237 non-null float64
    RBI                     7237 non-null float64
    SB                      7237 non-null float64
    CS                      7237 non-null float64
    BB                      7237 non-null float64
    SO                      7237 non-null float64
    IBB                     7237 non-null float64
    HBP                     7237 non-null float64
    SH                      7237 non-null float64
    SF                      7237 non-null float64
    A                       7237 non-null float64
    E                       7237 non-null float64
    DP                      7237 non-null float64
    years_allstar           7237 non-null float64
    Gold Glove              7237 non-null float64
    Most Valuable Player    7237 non-null float64
    Rookie of the Year      7237 non-null float64
    Silver Slugger          7237 non-null float64
    World Series MVP        7237 non-null float64
    HoF                     7237 non-null float64
    1901-1920               7237 non-null float64
    1921-1941               7237 non-null float64
    1942-1945               7237 non-null float64
    1946-1962               7237 non-null float64
    1963-1976               7237 non-null float64
    1977-1992               7237 non-null float64
    1993-2009               7237 non-null float64
    post2009                7237 non-null float64
    years_playing           7237 non-null float64
    G_all                   7237 non-null int64
    GS_percent              7237 non-null float64
    G_batting_percent       7237 non-null float64
    G_defense_percent       7237 non-null float64
    G_1b_percent            7237 non-null float64
    G_2b_percent            7237 non-null float64
    G_3b_percent            7237 non-null float64
    G_ss_percent            7237 non-null float64
    G_lf_percent            7237 non-null float64
    G_cf_percent            7237 non-null float64
    G_rf_percent            7237 non-null float64
    G_of_percent            7237 non-null float64
    G_dh_percent            7237 non-null float64
    dtypes: float64(46), int64(4), object(3)
    memory usage: 3.0+ MB


The only column that has NA values is `nameFirst`, and since there are only two of them, let's not worry about these. We have finally consolidated everything into a single DataFrame with everything we need to know about the players. In the next step, we are going to draw some insights from the data by adding new features.


```python
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>playerID</th>
      <th>nameFirst</th>
      <th>nameLast</th>
      <th>bats_R</th>
      <th>throws_R</th>
      <th>finalGame</th>
      <th>AB</th>
      <th>R</th>
      <th>H</th>
      <th>2B</th>
      <th>...</th>
      <th>G_defense_percent</th>
      <th>G_1b_percent</th>
      <th>G_2b_percent</th>
      <th>G_3b_percent</th>
      <th>G_ss_percent</th>
      <th>G_lf_percent</th>
      <th>G_cf_percent</th>
      <th>G_rf_percent</th>
      <th>G_of_percent</th>
      <th>G_dh_percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>aaronha01</td>
      <td>Hank</td>
      <td>Aaron</td>
      <td>1</td>
      <td>1</td>
      <td>1976</td>
      <td>12364.0</td>
      <td>2174.0</td>
      <td>3771.0</td>
      <td>624.0</td>
      <td>...</td>
      <td>0.905094</td>
      <td>0.063675</td>
      <td>0.013038</td>
      <td>0.002122</td>
      <td>0.000000</td>
      <td>0.095512</td>
      <td>0.093390</td>
      <td>0.659187</td>
      <td>0.836871</td>
      <td>0.060946</td>
    </tr>
    <tr>
      <th>1</th>
      <td>aaronto01</td>
      <td>Tommie</td>
      <td>Aaron</td>
      <td>1</td>
      <td>1</td>
      <td>1971</td>
      <td>944.0</td>
      <td>102.0</td>
      <td>216.0</td>
      <td>42.0</td>
      <td>...</td>
      <td>0.791762</td>
      <td>0.530892</td>
      <td>0.016018</td>
      <td>0.022883</td>
      <td>0.000000</td>
      <td>0.308924</td>
      <td>0.002288</td>
      <td>0.004577</td>
      <td>0.313501</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>abadan01</td>
      <td>Andy</td>
      <td>Abad</td>
      <td>0</td>
      <td>0</td>
      <td>2006</td>
      <td>21.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.600000</td>
      <td>0.533333</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.066667</td>
      <td>0.066667</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>abbated01</td>
      <td>Ed</td>
      <td>Abbaticchio</td>
      <td>1</td>
      <td>1</td>
      <td>1910</td>
      <td>3044.0</td>
      <td>355.0</td>
      <td>772.0</td>
      <td>99.0</td>
      <td>...</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.490058</td>
      <td>0.023392</td>
      <td>0.453801</td>
      <td>0.000000</td>
      <td>0.002339</td>
      <td>0.001170</td>
      <td>0.003509</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>abbotje01</td>
      <td>Jeff</td>
      <td>Abbott</td>
      <td>1</td>
      <td>0</td>
      <td>2001</td>
      <td>596.0</td>
      <td>82.0</td>
      <td>157.0</td>
      <td>33.0</td>
      <td>...</td>
      <td>0.793991</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.270386</td>
      <td>0.347639</td>
      <td>0.236052</td>
      <td>0.793991</td>
      <td>0.051502</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 53 columns</p>
</div>



## Feature Engineering

We'll start by adding important baseball statistics such as batting average, on-base percentage, slugging percentage, and on-base plus slugging percentage, using the following formulas:

* Batting Ave. = Hits / At Bats
* Plate Appearances = At Bats + Walks + Sacrifice Flys & Hits + Hit by Pitch
* On-base = (Hits + Walks + Hit by Pitch) / Plate Appearances
* Slugging = ((Home Runs x 4) + (Triples x 3) + (Doubles x 2) + Singles) / At Bats
* On-Base plus Slugging = On-base + Slugging

Since we are computing a lot of ratios, NA values may come about which we'll have to remove 


```python
# Create Batting Average (`AVE`) column
df['AVE'] = df['H'] / df['AB']

# Create On Base Percent (`OBP`) column
plate_appearances = (df['AB'] + df['BB'] + df['SF'] + df['SH'] + df['HBP'])
df['OBP'] = (df['H'] + df['BB'] + df['HBP']) / plate_appearances

# Create Slugging Percent (`Slug_Percent`) column
single = ((df['H'] - df['2B']) - df['3B']) - df['HR']
df['Slug_Percent'] = ((df['HR'] * 4) + (df['3B'] * 3) + (df['2B'] * 2) + single) / df['AB']

# Create On Base plus Slugging Percent (`OPS`) column
hr = df['HR'] * 4
triple = df['3B'] * 3
double = df['2B'] * 2
df['OPS'] = df['OBP'] + df['Slug_Percent']
```


```python
df = df.dropna()
print(df.isnull().sum(axis=0).tolist())
```

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


Yass! Now we are free of the NA plague. Before we move on to create some plots, let's try to identify the outliers in our data. They are players who boasted HoF-worthy stats but were ignored by HoF voters due to match-fixing scandals and performance enhancing drugs (PED) allegations. To find out who they were, we need to read up on the history of MLB. The following articles contain may help us in that regard:

* [List of people banned from Major League Baseball](https://en.wikipedia.org/wiki/List_of_people_banned_from_Major_League_Baseball)
* [These 11 Players' Hall of Fame Inductions Have Been Sabotaged by Steroid Allegations and Admissions](http://www.complex.com/sports/2016/07/players-hall-of-fame-inductions-destroyed-steroid-allegations/)
* [Top 15 Baseball Players Who Have Used Performance Enhancing Drugs](http://www.thesportster.com/baseball/top-15-baseball-players-who-have-used-performance-enhancing-drugs/)

Once we have done our homework, it's time to remove these names from our data.


```python
players = ['jacksjo01', 'rosepe01', 'giambja01', 'sheffga01', 'braunry02', 'bondsba01', \
           'palmera01', 'mcgwima01', 'clemero02', 'sosasa01', 'rodrial01']

df = df[~df['playerID'].isin(players)]
```

Next, we will plot out the distributions for certain statistics such as Hits, Home Runs, Years Playing, and Years Featured in All Star Game for Hall of Fame players to see if there are any trends among them.


```python
# Filter players who are in HoF
df_hof = df[df['HoF'] == 1]

print(len(df_hof))

sns.set(style="white", palette="muted", color_codes=True)

# Initialize the figure and add subplots
fig = plt.figure(figsize=(16, 14))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

# Create distribution plots for Hits, Home Runs, Years Played and All Star Games
sns.distplot(df_hof['H'], ax = ax1, kde = True, axlabel = False, color = 'r')
ax1.set_title('Distribution of Hits')
sns.distplot(df_hof['HR'], ax = ax2, kde = True, axlabel = False, color = 'r')
ax2.set_title('Distribution of Home Runs')
sns.distplot(df_hof['years_playing'], ax = ax3, kde = False, axlabel = False, color = 'r')
ax3.set_title('Distribution of Years Playing')
ax3.set_ylabel('HoF Careers')
sns.distplot(df_hof['years_allstar'], ax = ax4, kde = False, axlabel = False, color = 'r')
ax4.set_title('Distribution of Years Featured in All Star Game')
```

    72





    <matplotlib.text.Text at 0x17dcdcbe0>




![png]({{ site.url }}/assets/article_images/2018-07-08-predict-mlb-hof/p2-investigate-data-set_52_2.png)


We have 70 Hall of Famers in our data and they all boast admirable statistics. A few points to note:
* High number of Hits seem to be favorable: Most HoF players scored on average 3000 hits.
* Home Run is not so important: The majority of inductees didn't hit more than 200 home runs in their career.
* With experience comes votes: Players who have competed in more than 20 seasons make up a large portion of Hall of Famers.
* All Star Game appearances don't have much weight: In fact most players inducted only have participated in less than 10 games.

Now let's see how they fare against non-HoF players. To ensure we are comparing apples to apples, let's exclude non-HoF players with less than 10 years of experience.


```python
# Filter `df` for players with 10 or more years of experience
df_10 = df[(df['years_playing'] >= 10) & (df['HoF'] == 0)]

print(len(df_10))

# Initialize the figure and add subplots
fig = plt.figure(figsize=(16, 14))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

# Create distribution plots for Hits, Home Runs, Years Played and All Star Games
sns.distplot(df_10['H'], ax = ax1, kde = True, axlabel = False, bins = 12)
ax1.set_title('Distribution of Hits')
sns.distplot(df_10['HR'], ax = ax2, kde = True, axlabel = False, bins = 8)
ax2.set_title('Distribution of Home Runs')
sns.distplot(df_10['years_playing'], ax = ax3, kde = False, axlabel = False, bins = 10)
ax3.set_title('Distribution of Years Playing')
ax3.set_ylabel('HoF Careers')
sns.distplot(df_10['years_allstar'], ax = ax4, kde = False, axlabel = False, bins = 8)
ax4.set_title('Distribution of Years Featured in All Star Game')
```

    1665





    <matplotlib.text.Text at 0x17f197710>




![png]({{ site.url }}/assets/article_images/2018-07-08-predict-mlb-hof/p2-investigate-data-set_54_2.png)


There are 1675 non-HoF players in our data and it's fair to say that most of them are less experienced players, which partly explains their lackluster statistics compared to the veterans who have made it to Cooperstown.

Next we want to see how Hits vs. Batting Average and Home Runs vs. Batting Average differ between HoF and non-HoF players.


```python
# Initialize the figure and add subplots
fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

# Create Scatter plots for Hits vs. Average and Home Runs vs. Average
ax1.scatter(df_hof['H'], df_hof['AVE'], c='r', label='HoF Player')
ax1.scatter(df_10['H'], df_10['AVE'], c='b', label='Non HoF Player')
ax1.set_title('Career Hits vs. Career Batting Average')
ax1.set_xlabel('Career Hits')
ax1.set_ylabel('Career Average')
ax2.scatter(df_hof['HR'], df_hof['AVE'], c='r', label='HoF Player')
ax2.scatter(df_10['HR'], df_10['AVE'], c='b', label='Non HoF Player')
ax2.set_title('Career Home Runs vs. Career Batting Average')
ax2.set_xlabel('Career Home Runs')
ax2.legend(loc='lower right', scatterpoints=1)

```




    <matplotlib.legend.Legend at 0x1834a5f28>




![png]({{ site.url }}/assets/article_images/2018-07-08-predict-mlb-hof/p2-investigate-data-set_56_1.png)


Suffice to say, it's not surprising to see HoF players as high-achievers compared to their non-HoF teammates. There seem to be a positive correlation between career hits and career batting average regardless of HoF status, however the relationship is not as strong when it comes to home runs versus batting average. 

With this we have answered the first question posed at the beginning of this post. To answer the second one, we need to build some machine learning model to predict whether an eligible player will ever be elected to the HoF. 

## Preparing Training and Test Data

Since a player must wait 5 years to become eligible for the HoF ballot, and can remain on the ballot for as many as 10 years then there are still eligible players who played their final season in the last 15 years. Hence those who played their last games in 2003 will be eligible for consideration in 2018 and so on. 


```python
# Filter `df` for players who retired more than 15 years ago
df_hitters = df[df['finalGame'] < 2002]

# Filter `df` for players who retired less than 15 years ago and for currently active players
df_eligible = df[df['finalGame'] >= 2002]

# Players who retired less than 15 years ago but more than 5 years ago and were inducted
early_inductees = df_eligible[df_eligible['HoF'] == 1]

# Remove these players from `df_eligible`
df_eligible = df_eligible[df_eligible['HoF'] != 1]

# Add these players to `df_hitters`
df_hitters = df_hitters.append(early_inductees)
```

`df_hitters` is what we will use to train and test our model on since it contains statistics of past Hall of Famers while `df_eligible` is the "new" data consisting of eligible players that we would like to make predictions of.


```python
print(len(df_hitters))

# Separate `df_hitters` into target (response) and features (predictors)
target = df_hitters['HoF']
features = df_hitters.drop(['playerID', 'nameFirst', 'nameLast', 'HoF'], axis=1)
```

    5487


## Logistic Regression

The first model we'll try is a Logistic Regression model and we'll be using the Kfold cross-validation technique.


```python
from sklearn.cross_validation import cross_val_predict, KFold
from sklearn.linear_model import LogisticRegression

# Create Logistic Regression model
lr = LogisticRegression(class_weight='balanced')

# Create an instance of the KFold class
kf = KFold(features.shape[0], random_state=1)

# Create predictions using cross validation
predictions_lr = cross_val_predict(lr, features, target, cv=kf)
```

To determine accuracy, we need to compare our predictions to the target. The error metrics we'll be using are counts and rates of True Positive (TP), False Positive (FP), and False Negative (FN), whose definitions are given below:

* True Positive: The player was predicted to be in the HoF and they are a HoF member.
* False Positive: The player was predicted to be in the HoF but they are not a HoF member.
* False Negative: The player was predicted to be not in the HoF but they are indeed a HoF member.
* True Negative: The player was predicted not to be in the HoF and they are not a HoF member.

From here, we can compute the rates as follows:

* True Positive rate: # True Positive / (# True Positive + # False Negative)
* False Negative rate: # False Negative / (# False Negative + # True Positive)
* False Positive rate: # False Positive / (# False Positive + # True Negative)


```python
# Import NumPy as np
import numpy as np

# Convert predictions and target to NumPy arrays
np_predictions_lr = np.asarray(predictions_lr)
np_target = target.as_matrix()
```


```python
# Create a function to report TP, FP, and FN rates
def predAccuracy(predictions, target):
    
    # Determine True Positive count
    tp_filter = (predictions == 1) & (target == 1)
    tp = len(predictions[tp_filter])

    # Determine False Negative count
    fn_filter = (predictions == 0) & (target == 1)
    fn = len(predictions[fn_filter])

    # Determine False Positive count
    fp_filter = (predictions == 1) & (target == 0)
    fp = len(predictions[fp_filter])

    # Determine True Negative count
    tn_filter = (predictions == 0) & (target == 0)
    tn = len(predictions[tn_filter])

    # Determine True Positive rate
    tpr = tp / (tp + fn)

    # Determine False Negative rate
    fnr = fn / (fn_lr + tp)

    # Determine False Positive rate
    fpr = fp / (fp + tn)

    # Print each count
    print("True Positive Count: {0}".format(tp))
    print("False Negative Count: {0}".format(fn))
    print("False Positive Count: {0}".format(fp))

    # Print each rate
    print("True Positive Rate: {0:6.4f}".format(tpr))
    print("False Negative Rate: {0:6.4f}".format(fnr))
    print("False Positive Rate: {0:6.4f}".format(fpr))
```


```python
# Accuracy rates of logistic regression model
predAccuracy(np_predictions_lr, np_target)
```

    True Positive Count: 60
    False Negative Count: 12
    False Positive Count: 35
    True Positive Rate: 0.8333
    False Negative Rate: 0.1667
    False Positive Rate: 0.0065


## Random Forest

What we're trying to answer is a classic example of a classification problem, and it would be a crime not to mention random forest algorithm at some point. In the following we'll see how this algorithm stacks up against the logistic regression model.


```python
# Import RandomForestClassifier from sklearn
from sklearn.ensemble import RandomForestClassifier

# Create penalty dictionary
penalty = {
    0: 100,
    1: 1
}

# Create Random Forest model
rf = RandomForestClassifier(random_state=1,n_estimators=12, max_depth=11, min_samples_leaf=1, class_weight=penalty)

# Create predictions using cross validation
predictions_rf = cross_val_predict(rf, features, target, cv=kf)

# Convert predictions to NumPy array
np_predictions_rf = np.asarray(predictions_rf)
```


```python
# Accuracy rates of random forest model
predAccuracy(np_predictions_rf, np_target)
```

    True Positive Count: 51
    False Negative Count: 21
    False Positive Count: 8
    True Positive Rate: 0.7083
    False Negative Rate: 0.3333
    False Positive Rate: 0.0015


Although the random forest is less accurate, predicting only 51 of 72 Hall of Famers, its FN and FP counts are far fewer. Hence it will be the model of choice to make our predictions.

## Making predictions

We'll use the trained and tested random forest model to make predictions on the probability of getting voted into the HoF for each player in `df_eligible` and then print out 50 players who have the highest chance of doing so. This will also answer our second question.


```python
# Create a new features DataFrame
new_features = df_eligible.drop(['playerID', 'nameFirst', 'nameLast', 'HoF'], axis=1)

# Fit the Random Forest model
rf.fit(features, target)

# Estimate probabilities of Hall of Fame induction
probabilities = rf.predict_proba(new_features)

# Convert predictions to a DataFrame
hof_predictions = pd.DataFrame(probabilities[:,1])

# Sort the DataFrame (descending)
hof_predictions = hof_predictions.sort_values(0, ascending=False)
hof_predictions.rename(columns = {0:'prob'}, inplace = True)

# Merge the prediction with new_data
new_data.index = range(len(new_data))
new_data.head()
hof_predictions = hof_predictions.join(new_data, how = 'left')
hof_predictions.index = range(len(hof_predictions))
hof_predictions.head(50)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>prob</th>
      <th>playerID</th>
      <th>nameFirst</th>
      <th>nameLast</th>
      <th>bats_R</th>
      <th>throws_R</th>
      <th>finalGame</th>
      <th>AB</th>
      <th>R</th>
      <th>H</th>
      <th>...</th>
      <th>G_ss_percent</th>
      <th>G_lf_percent</th>
      <th>G_cf_percent</th>
      <th>G_rf_percent</th>
      <th>G_of_percent</th>
      <th>G_dh_percent</th>
      <th>AVE</th>
      <th>OBP</th>
      <th>Slug_Percent</th>
      <th>OPS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>ortizda01</td>
      <td>David</td>
      <td>Ortiz</td>
      <td>0</td>
      <td>0</td>
      <td>2016</td>
      <td>8640.0</td>
      <td>1419.0</td>
      <td>2472.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.842608</td>
      <td>0.286111</td>
      <td>0.379447</td>
      <td>0.551505</td>
      <td>0.930952</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.000000</td>
      <td>ramirma02</td>
      <td>Manny</td>
      <td>Ramirez</td>
      <td>1</td>
      <td>1</td>
      <td>2011</td>
      <td>8244.0</td>
      <td>1544.0</td>
      <td>2574.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.450478</td>
      <td>0.000000</td>
      <td>0.392702</td>
      <td>0.841877</td>
      <td>0.144222</td>
      <td>0.312227</td>
      <td>0.410477</td>
      <td>0.585395</td>
      <td>0.995872</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.000000</td>
      <td>cabremi01</td>
      <td>Miguel</td>
      <td>Cabrera</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
      <td>7853.0</td>
      <td>1321.0</td>
      <td>2519.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.118321</td>
      <td>0.000000</td>
      <td>0.047710</td>
      <td>0.165553</td>
      <td>0.037214</td>
      <td>0.320769</td>
      <td>0.398511</td>
      <td>0.562078</td>
      <td>0.960589</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.000000</td>
      <td>pujolal01</td>
      <td>Albert</td>
      <td>Pujols</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
      <td>9138.0</td>
      <td>1670.0</td>
      <td>2825.0</td>
      <td>...</td>
      <td>0.000412</td>
      <td>0.110882</td>
      <td>0.000000</td>
      <td>0.016488</td>
      <td>0.127370</td>
      <td>0.139736</td>
      <td>0.309149</td>
      <td>0.392248</td>
      <td>0.572554</td>
      <td>0.964802</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.000000</td>
      <td>jeterde01</td>
      <td>Derek</td>
      <td>Jeter</td>
      <td>1</td>
      <td>1</td>
      <td>2014</td>
      <td>11195.0</td>
      <td>1923.0</td>
      <td>3465.0</td>
      <td>...</td>
      <td>0.973426</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.026574</td>
      <td>0.309513</td>
      <td>0.374306</td>
      <td>0.439571</td>
      <td>0.813877</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.000000</td>
      <td>jonesch06</td>
      <td>Chipper</td>
      <td>Jones</td>
      <td>0</td>
      <td>1</td>
      <td>2012</td>
      <td>8984.0</td>
      <td>1619.0</td>
      <td>2726.0</td>
      <td>...</td>
      <td>0.019608</td>
      <td>0.142457</td>
      <td>0.000000</td>
      <td>0.003601</td>
      <td>0.145658</td>
      <td>0.011204</td>
      <td>0.303428</td>
      <td>0.400980</td>
      <td>0.529274</td>
      <td>0.930254</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1.000000</td>
      <td>vizquom01</td>
      <td>Omar</td>
      <td>Vizquel</td>
      <td>0</td>
      <td>1</td>
      <td>2012</td>
      <td>10586.0</td>
      <td>1445.0</td>
      <td>2877.0</td>
      <td>...</td>
      <td>0.912736</td>
      <td>0.000337</td>
      <td>0.000000</td>
      <td>0.000337</td>
      <td>0.000674</td>
      <td>0.002358</td>
      <td>0.271774</td>
      <td>0.329143</td>
      <td>0.352069</td>
      <td>0.681212</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.000000</td>
      <td>heltoto01</td>
      <td>Todd</td>
      <td>Helton</td>
      <td>0</td>
      <td>0</td>
      <td>2013</td>
      <td>7962.0</td>
      <td>1401.0</td>
      <td>2519.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.005785</td>
      <td>0.000000</td>
      <td>0.000890</td>
      <td>0.006676</td>
      <td>0.000890</td>
      <td>0.316378</td>
      <td>0.413862</td>
      <td>0.539061</td>
      <td>0.952923</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1.000000</td>
      <td>abreubo01</td>
      <td>Bobby</td>
      <td>Abreu</td>
      <td>0</td>
      <td>1</td>
      <td>2014</td>
      <td>8480.0</td>
      <td>1453.0</td>
      <td>2470.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.058557</td>
      <td>0.008660</td>
      <td>0.820619</td>
      <td>0.881649</td>
      <td>0.066392</td>
      <td>0.291274</td>
      <td>0.394703</td>
      <td>0.474764</td>
      <td>0.869467</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1.000000</td>
      <td>beltrad01</td>
      <td>Adrian</td>
      <td>Beltre</td>
      <td>1</td>
      <td>1</td>
      <td>2016</td>
      <td>10295.0</td>
      <td>1428.0</td>
      <td>2942.0</td>
      <td>...</td>
      <td>0.002574</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.030882</td>
      <td>0.285770</td>
      <td>0.337833</td>
      <td>0.479845</td>
      <td>0.817678</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.000000</td>
      <td>gonzalu01</td>
      <td>Luis</td>
      <td>Gonzalez</td>
      <td>0</td>
      <td>1</td>
      <td>2008</td>
      <td>9157.0</td>
      <td>1412.0</td>
      <td>2591.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.933230</td>
      <td>0.003474</td>
      <td>0.008491</td>
      <td>0.942107</td>
      <td>0.010807</td>
      <td>0.282953</td>
      <td>0.366252</td>
      <td>0.478869</td>
      <td>0.845121</td>
    </tr>
    <tr>
      <th>11</th>
      <td>0.916705</td>
      <td>thomeji01</td>
      <td>Jim</td>
      <td>Thome</td>
      <td>0</td>
      <td>1</td>
      <td>2012</td>
      <td>8422.0</td>
      <td>1583.0</td>
      <td>2328.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.321667</td>
      <td>0.276419</td>
      <td>0.401823</td>
      <td>0.554144</td>
      <td>0.955967</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.916667</td>
      <td>kentje01</td>
      <td>Jeff</td>
      <td>Kent</td>
      <td>1</td>
      <td>1</td>
      <td>2008</td>
      <td>8498.0</td>
      <td>1320.0</td>
      <td>2461.0</td>
      <td>...</td>
      <td>0.001305</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.003046</td>
      <td>0.289598</td>
      <td>0.355143</td>
      <td>0.499647</td>
      <td>0.854790</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.916667</td>
      <td>guerrvl01</td>
      <td>Vladimir</td>
      <td>Guerrero</td>
      <td>1</td>
      <td>1</td>
      <td>2011</td>
      <td>8155.0</td>
      <td>1328.0</td>
      <td>2590.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000466</td>
      <td>0.000932</td>
      <td>0.747555</td>
      <td>0.748952</td>
      <td>0.236609</td>
      <td>0.317597</td>
      <td>0.378629</td>
      <td>0.552544</td>
      <td>0.931173</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.833417</td>
      <td>gonzaju03</td>
      <td>Juan</td>
      <td>Gonzalez</td>
      <td>1</td>
      <td>1</td>
      <td>2005</td>
      <td>6556.0</td>
      <td>1061.0</td>
      <td>1936.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.216696</td>
      <td>0.149201</td>
      <td>0.449378</td>
      <td>0.776199</td>
      <td>0.217880</td>
      <td>0.295302</td>
      <td>0.343117</td>
      <td>0.560708</td>
      <td>0.903824</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.833333</td>
      <td>beltrca01</td>
      <td>Carlos</td>
      <td>Beltran</td>
      <td>0</td>
      <td>1</td>
      <td>2016</td>
      <td>9301.0</td>
      <td>1522.0</td>
      <td>2617.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000814</td>
      <td>0.639805</td>
      <td>0.256410</td>
      <td>0.893773</td>
      <td>0.082621</td>
      <td>0.281368</td>
      <td>0.353165</td>
      <td>0.491560</td>
      <td>0.844725</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.750093</td>
      <td>walkela01</td>
      <td>Larry</td>
      <td>Walker</td>
      <td>0</td>
      <td>1</td>
      <td>2005</td>
      <td>6907.0</td>
      <td>1355.0</td>
      <td>2160.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.016600</td>
      <td>0.034708</td>
      <td>0.864185</td>
      <td>0.907445</td>
      <td>0.013581</td>
      <td>0.312726</td>
      <td>0.399875</td>
      <td>0.565224</td>
      <td>0.965099</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.750093</td>
      <td>camermi01</td>
      <td>Mike</td>
      <td>Cameron</td>
      <td>1</td>
      <td>1</td>
      <td>2011</td>
      <td>6839.0</td>
      <td>1064.0</td>
      <td>1700.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.003069</td>
      <td>0.914066</td>
      <td>0.083887</td>
      <td>0.982609</td>
      <td>0.005627</td>
      <td>0.248574</td>
      <td>0.336631</td>
      <td>0.443778</td>
      <td>0.780409</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.750079</td>
      <td>tejadmi01</td>
      <td>Miguel</td>
      <td>Tejada</td>
      <td>1</td>
      <td>1</td>
      <td>2013</td>
      <td>8434.0</td>
      <td>1230.0</td>
      <td>2407.0</td>
      <td>...</td>
      <td>0.896361</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012437</td>
      <td>0.285392</td>
      <td>0.334891</td>
      <td>0.455537</td>
      <td>0.790428</td>
    </tr>
    <tr>
      <th>19</th>
      <td>0.750079</td>
      <td>ramirar01</td>
      <td>Aramis</td>
      <td>Ramirez</td>
      <td>1</td>
      <td>1</td>
      <td>2015</td>
      <td>8136.0</td>
      <td>1098.0</td>
      <td>2303.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.011851</td>
      <td>0.283063</td>
      <td>0.340864</td>
      <td>0.492134</td>
      <td>0.832997</td>
    </tr>
    <tr>
      <th>20</th>
      <td>0.750076</td>
      <td>ordonma01</td>
      <td>Magglio</td>
      <td>Ordonez</td>
      <td>1</td>
      <td>1</td>
      <td>2011</td>
      <td>6978.0</td>
      <td>1076.0</td>
      <td>2156.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.014610</td>
      <td>0.926948</td>
      <td>0.933442</td>
      <td>0.056277</td>
      <td>0.308971</td>
      <td>0.368380</td>
      <td>0.502436</td>
      <td>0.870816</td>
    </tr>
    <tr>
      <th>21</th>
      <td>0.750070</td>
      <td>renteed01</td>
      <td>Edgar</td>
      <td>Renteria</td>
      <td>1</td>
      <td>1</td>
      <td>2011</td>
      <td>8142.0</td>
      <td>1200.0</td>
      <td>2327.0</td>
      <td>...</td>
      <td>0.982342</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000929</td>
      <td>0.285802</td>
      <td>0.339290</td>
      <td>0.398059</td>
      <td>0.737349</td>
    </tr>
    <tr>
      <th>22</th>
      <td>0.750070</td>
      <td>damonjo01</td>
      <td>Johnny</td>
      <td>Damon</td>
      <td>0</td>
      <td>0</td>
      <td>2012</td>
      <td>9736.0</td>
      <td>1668.0</td>
      <td>2769.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.274699</td>
      <td>0.521285</td>
      <td>0.059036</td>
      <td>0.830120</td>
      <td>0.149398</td>
      <td>0.284408</td>
      <td>0.350096</td>
      <td>0.432827</td>
      <td>0.782923</td>
    </tr>
    <tr>
      <th>23</th>
      <td>0.750070</td>
      <td>jonesan01</td>
      <td>Andruw</td>
      <td>Jones</td>
      <td>1</td>
      <td>1</td>
      <td>2012</td>
      <td>7599.0</td>
      <td>1204.0</td>
      <td>1933.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.051002</td>
      <td>0.785064</td>
      <td>0.102004</td>
      <td>0.930328</td>
      <td>0.048725</td>
      <td>0.254376</td>
      <td>0.337142</td>
      <td>0.485590</td>
      <td>0.822732</td>
    </tr>
    <tr>
      <th>24</th>
      <td>0.750070</td>
      <td>anderga01</td>
      <td>Garret</td>
      <td>Anderson</td>
      <td>0</td>
      <td>0</td>
      <td>2010</td>
      <td>8640.0</td>
      <td>1084.0</td>
      <td>2529.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.622531</td>
      <td>0.181329</td>
      <td>0.072711</td>
      <td>0.858618</td>
      <td>0.106373</td>
      <td>0.292708</td>
      <td>0.323199</td>
      <td>0.461111</td>
      <td>0.784310</td>
    </tr>
    <tr>
      <th>25</th>
      <td>0.750061</td>
      <td>delgaca01</td>
      <td>Carlos</td>
      <td>Delgado</td>
      <td>0</td>
      <td>1</td>
      <td>2009</td>
      <td>7283.0</td>
      <td>1241.0</td>
      <td>2038.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.028501</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.028501</td>
      <td>0.090909</td>
      <td>0.279830</td>
      <td>0.383389</td>
      <td>0.545929</td>
      <td>0.929318</td>
    </tr>
    <tr>
      <th>26</th>
      <td>0.750056</td>
      <td>suzukic01</td>
      <td>Ichiro</td>
      <td>Suzuki</td>
      <td>0</td>
      <td>1</td>
      <td>2016</td>
      <td>9689.0</td>
      <td>1396.0</td>
      <td>3030.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.038800</td>
      <td>0.124800</td>
      <td>0.780400</td>
      <td>0.927200</td>
      <td>0.020800</td>
      <td>0.312726</td>
      <td>0.354481</td>
      <td>0.404583</td>
      <td>0.759064</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.750039</td>
      <td>francju01</td>
      <td>Julio</td>
      <td>Franco</td>
      <td>1</td>
      <td>1</td>
      <td>2007</td>
      <td>8677.0</td>
      <td>1285.0</td>
      <td>2586.0</td>
      <td>...</td>
      <td>0.282153</td>
      <td>0.001583</td>
      <td>0.000000</td>
      <td>0.000396</td>
      <td>0.001583</td>
      <td>0.148397</td>
      <td>0.298029</td>
      <td>0.363889</td>
      <td>0.417195</td>
      <td>0.781083</td>
    </tr>
    <tr>
      <th>28</th>
      <td>0.750038</td>
      <td>mcgrifr01</td>
      <td>Fred</td>
      <td>McGriff</td>
      <td>0</td>
      <td>0</td>
      <td>2004</td>
      <td>8757.0</td>
      <td>1349.0</td>
      <td>2490.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071138</td>
      <td>0.284344</td>
      <td>0.376843</td>
      <td>0.509078</td>
      <td>0.885921</td>
    </tr>
    <tr>
      <th>29</th>
      <td>0.666786</td>
      <td>leeca01</td>
      <td>Carlos</td>
      <td>Lee</td>
      <td>1</td>
      <td>1</td>
      <td>2012</td>
      <td>7983.0</td>
      <td>1125.0</td>
      <td>2273.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.843259</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.843259</td>
      <td>0.039066</td>
      <td>0.284730</td>
      <td>0.338607</td>
      <td>0.482776</td>
      <td>0.821383</td>
    </tr>
    <tr>
      <th>30</th>
      <td>0.666770</td>
      <td>willibe02</td>
      <td>Bernie</td>
      <td>Williams</td>
      <td>0</td>
      <td>1</td>
      <td>2006</td>
      <td>7869.0</td>
      <td>1366.0</td>
      <td>2336.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.004335</td>
      <td>0.894027</td>
      <td>0.029865</td>
      <td>0.926782</td>
      <td>0.062139</td>
      <td>0.296861</td>
      <td>0.380426</td>
      <td>0.477316</td>
      <td>0.857742</td>
    </tr>
    <tr>
      <th>31</th>
      <td>0.666769</td>
      <td>rolensc01</td>
      <td>Scott</td>
      <td>Rolen</td>
      <td>1</td>
      <td>1</td>
      <td>2012</td>
      <td>7398.0</td>
      <td>1211.0</td>
      <td>2077.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.280752</td>
      <td>0.364287</td>
      <td>0.490403</td>
      <td>0.854690</td>
    </tr>
    <tr>
      <th>32</th>
      <td>0.666769</td>
      <td>cabreor01</td>
      <td>Orlando</td>
      <td>Cabrera</td>
      <td>1</td>
      <td>1</td>
      <td>2011</td>
      <td>7562.0</td>
      <td>985.0</td>
      <td>2055.0</td>
      <td>...</td>
      <td>0.928463</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.002519</td>
      <td>0.271754</td>
      <td>0.315082</td>
      <td>0.389712</td>
      <td>0.704793</td>
    </tr>
    <tr>
      <th>33</th>
      <td>0.666761</td>
      <td>aloumo01</td>
      <td>Moises</td>
      <td>Alou</td>
      <td>1</td>
      <td>1</td>
      <td>2008</td>
      <td>7037.0</td>
      <td>1109.0</td>
      <td>2134.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.639547</td>
      <td>0.051493</td>
      <td>0.310505</td>
      <td>0.959835</td>
      <td>0.011843</td>
      <td>0.303254</td>
      <td>0.368887</td>
      <td>0.515703</td>
      <td>0.884589</td>
    </tr>
    <tr>
      <th>34</th>
      <td>0.666761</td>
      <td>durhara01</td>
      <td>Ray</td>
      <td>Durham</td>
      <td>0</td>
      <td>1</td>
      <td>2008</td>
      <td>7408.0</td>
      <td>1249.0</td>
      <td>2054.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000506</td>
      <td>0.000000</td>
      <td>0.000506</td>
      <td>0.027342</td>
      <td>0.277268</td>
      <td>0.349757</td>
      <td>0.435745</td>
      <td>0.785502</td>
    </tr>
    <tr>
      <th>35</th>
      <td>0.666761</td>
      <td>loftoke01</td>
      <td>Kenny</td>
      <td>Lofton</td>
      <td>0</td>
      <td>0</td>
      <td>2007</td>
      <td>8120.0</td>
      <td>1528.0</td>
      <td>2428.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.023776</td>
      <td>0.943414</td>
      <td>0.004755</td>
      <td>0.970518</td>
      <td>0.005706</td>
      <td>0.299015</td>
      <td>0.368746</td>
      <td>0.422783</td>
      <td>0.791529</td>
    </tr>
    <tr>
      <th>36</th>
      <td>0.666761</td>
      <td>martied01</td>
      <td>Edgar</td>
      <td>Martinez</td>
      <td>1</td>
      <td>1</td>
      <td>2004</td>
      <td>7213.0</td>
      <td>1219.0</td>
      <td>2247.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.682725</td>
      <td>0.311521</td>
      <td>0.417320</td>
      <td>0.515458</td>
      <td>0.932778</td>
    </tr>
    <tr>
      <th>37</th>
      <td>0.666760</td>
      <td>edmonji01</td>
      <td>Jim</td>
      <td>Edmonds</td>
      <td>0</td>
      <td>0</td>
      <td>2010</td>
      <td>6858.0</td>
      <td>1251.0</td>
      <td>1949.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.030830</td>
      <td>0.879165</td>
      <td>0.023869</td>
      <td>0.928394</td>
      <td>0.010443</td>
      <td>0.284194</td>
      <td>0.375439</td>
      <td>0.527122</td>
      <td>0.902560</td>
    </tr>
    <tr>
      <th>38</th>
      <td>0.666760</td>
      <td>leede02</td>
      <td>Derrek</td>
      <td>Lee</td>
      <td>1</td>
      <td>1</td>
      <td>2011</td>
      <td>6962.0</td>
      <td>1081.0</td>
      <td>1959.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.004119</td>
      <td>0.281385</td>
      <td>0.364561</td>
      <td>0.494685</td>
      <td>0.859247</td>
    </tr>
    <tr>
      <th>39</th>
      <td>0.666746</td>
      <td>hunteto01</td>
      <td>Torii</td>
      <td>Hunter</td>
      <td>1</td>
      <td>1</td>
      <td>2015</td>
      <td>8857.0</td>
      <td>1296.0</td>
      <td>2452.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.007167</td>
      <td>0.642074</td>
      <td>0.305228</td>
      <td>0.951096</td>
      <td>0.039629</td>
      <td>0.276843</td>
      <td>0.331201</td>
      <td>0.461443</td>
      <td>0.792644</td>
    </tr>
    <tr>
      <th>40</th>
      <td>0.666738</td>
      <td>olerujo01</td>
      <td>John</td>
      <td>Olerud</td>
      <td>0</td>
      <td>0</td>
      <td>2005</td>
      <td>7592.0</td>
      <td>1139.0</td>
      <td>2239.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.059534</td>
      <td>0.294916</td>
      <td>0.397440</td>
      <td>0.464963</td>
      <td>0.862403</td>
    </tr>
    <tr>
      <th>41</th>
      <td>0.666738</td>
      <td>grissma02</td>
      <td>Marquis</td>
      <td>Grissom</td>
      <td>1</td>
      <td>1</td>
      <td>2005</td>
      <td>8275.0</td>
      <td>1187.0</td>
      <td>2251.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.039261</td>
      <td>0.908545</td>
      <td>0.024480</td>
      <td>0.964434</td>
      <td>0.000924</td>
      <td>0.272024</td>
      <td>0.316442</td>
      <td>0.414502</td>
      <td>0.730943</td>
    </tr>
    <tr>
      <th>42</th>
      <td>0.666737</td>
      <td>grudzma01</td>
      <td>Mark</td>
      <td>Grudzielanek</td>
      <td>1</td>
      <td>1</td>
      <td>2010</td>
      <td>7052.0</td>
      <td>946.0</td>
      <td>2040.0</td>
      <td>...</td>
      <td>0.347392</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.001110</td>
      <td>0.289280</td>
      <td>0.330001</td>
      <td>0.393222</td>
      <td>0.723223</td>
    </tr>
    <tr>
      <th>43</th>
      <td>0.666737</td>
      <td>ibanera01</td>
      <td>Raul</td>
      <td>Ibanez</td>
      <td>0</td>
      <td>1</td>
      <td>2014</td>
      <td>7471.0</td>
      <td>1055.0</td>
      <td>2034.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.678852</td>
      <td>0.000925</td>
      <td>0.079130</td>
      <td>0.750116</td>
      <td>0.143915</td>
      <td>0.272253</td>
      <td>0.335347</td>
      <td>0.465132</td>
      <td>0.800479</td>
    </tr>
    <tr>
      <th>44</th>
      <td>0.666737</td>
      <td>konerpa01</td>
      <td>Paul</td>
      <td>Konerko</td>
      <td>1</td>
      <td>1</td>
      <td>2014</td>
      <td>8393.0</td>
      <td>1162.0</td>
      <td>2340.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.007663</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.007663</td>
      <td>0.146871</td>
      <td>0.278804</td>
      <td>0.354024</td>
      <td>0.486477</td>
      <td>0.840501</td>
    </tr>
    <tr>
      <th>45</th>
      <td>0.666729</td>
      <td>gilesbr02</td>
      <td>Brian</td>
      <td>Giles</td>
      <td>0</td>
      <td>0</td>
      <td>2009</td>
      <td>6527.0</td>
      <td>1121.0</td>
      <td>1897.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.349756</td>
      <td>0.164050</td>
      <td>0.479155</td>
      <td>0.956145</td>
      <td>0.021657</td>
      <td>0.290639</td>
      <td>0.399617</td>
      <td>0.502375</td>
      <td>0.901992</td>
    </tr>
    <tr>
      <th>46</th>
      <td>0.666706</td>
      <td>finlest01</td>
      <td>Steve</td>
      <td>Finley</td>
      <td>0</td>
      <td>0</td>
      <td>2007</td>
      <td>9397.0</td>
      <td>1443.0</td>
      <td>2548.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.013937</td>
      <td>0.895858</td>
      <td>0.073171</td>
      <td>0.962447</td>
      <td>0.005420</td>
      <td>0.271150</td>
      <td>0.329350</td>
      <td>0.442375</td>
      <td>0.771725</td>
    </tr>
    <tr>
      <th>47</th>
      <td>0.666673</td>
      <td>polanpl01</td>
      <td>Placido</td>
      <td>Polanco</td>
      <td>1</td>
      <td>1</td>
      <td>2013</td>
      <td>7214.0</td>
      <td>1009.0</td>
      <td>2142.0</td>
      <td>...</td>
      <td>0.063311</td>
      <td>0.002595</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.002595</td>
      <td>0.001557</td>
      <td>0.296923</td>
      <td>0.338828</td>
      <td>0.397283</td>
      <td>0.736111</td>
    </tr>
    <tr>
      <th>48</th>
      <td>0.666667</td>
      <td>kotsama01</td>
      <td>Mark</td>
      <td>Kotsay</td>
      <td>0</td>
      <td>0</td>
      <td>2013</td>
      <td>6464.0</td>
      <td>790.0</td>
      <td>1784.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.032393</td>
      <td>0.528736</td>
      <td>0.242424</td>
      <td>0.792059</td>
      <td>0.030825</td>
      <td>0.275990</td>
      <td>0.330708</td>
      <td>0.404394</td>
      <td>0.735101</td>
    </tr>
    <tr>
      <th>49</th>
      <td>0.666667</td>
      <td>matthga02</td>
      <td>Gary</td>
      <td>Matthews</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>4103.0</td>
      <td>612.0</td>
      <td>1056.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.141296</td>
      <td>0.583138</td>
      <td>0.227166</td>
      <td>0.903201</td>
      <td>0.021077</td>
      <td>0.257373</td>
      <td>0.330951</td>
      <td>0.405313</td>
      <td>0.736264</td>
    </tr>
  </tbody>
</table>
<p>50 rows × 58 columns</p>
</div>



## Limitations

While it's nice to be able to tell who are likely to get elected in future ballots, this is a naïve model: we don’t predict the career trajectories of current players. We simply ask: if they retired now, and we relax the 10-year minimum requirement, would their statistical output qualify them for the Hall of Fame based on what we’ve seen voters do in the past?

This brings us to a related question: how much would steroid suspicions hurt the chance of getting in? Take Ivan Rodriguez for example. Despite allegations of injections of PED in 2003, he nevertheless made it in to the HoF in 2016. Along with cases for Tim Raines and Jeff Bagwell, who were also suspected of steroid use, this proves voters do forgive, but on what conditions and to what extent we never know. 

## References

1. [Lahman's Baseball Database](http://www.seanlahman.com/baseball-archive/statistics/)
2. [Scikit-Learn Tutorial: Baseball Analytics in Python Pt 2](https://www.datacamp.com/community/tutorials/scikit-learn-tutorial-baseball-2)
3. [Hall of Famers - Rules for Election](http://baseballhall.org/hall-of-famers/bbwaa-rules-for-election)
4. [What Are the Major Eras of Major League Baseball History?](http://www.huffingtonpost.com/quora/what-are-the-major-eras-o_b_3547814.html)
5. [List of people banned from Major League Baseball](https://en.wikipedia.org/wiki/List_of_people_banned_from_Major_League_Baseball)
6. [These 11 Players' Hall of Fame Inductions Have Been Sabotaged by Steroid Allegations and Admissions](http://www.complex.com/sports/2016/07/players-hall-of-fame-inductions-destroyed-steroid-allegations/)
7. [Top 15 Baseball Players Who Have Used Performance Enhancing Drugs](http://www.thesportster.com/baseball/top-15-baseball-players-who-have-used-performance-enhancing-drugs/)
8. [Hall of Fame Classification Using Random Forest](https://baseballwithr.wordpress.com/2014/11/26/hall-of-fame-classification-using-randomforest/)


```python

```
