# March-Madness-Bracket-Prediction
Code that uses random forest regression to predict a march madness bracket

There are two parts of the code: the datasetup and the model training/bracket predicting parts.

The datasetup part uses scraped boxscore data to set up a dataframe wherein each row has boxscore statistics from the teams previous games to predict the score of the current game. These statistics are in reference to the team that they're playing so if team1 has a win% of .5 and team2 has a win% of .75, the row that has features trying to predict team1's score would have win% of .5 - .75 = -.25. For each boxscore statistic theres a column that has the mean and a column with the standard deviation of it.

It takes ~50 minutes to set up a seasons dataframe due to recursive for loops it isn't efficient at all, so I included 2019-2024 season dataframes that are already set up and ready to be put into the model.

The model training/bracket predicting part of the code uses a random forest regression model or a neural network to see how they compare with each other. The hyperparameters haven't been optimized with grid search/optuna so there's still potential to do that. For any bracket besides 2024, the march madness teams will have to be manually entered into the mm_teams list. The way to enter teams is start on the top left part of the bracket and work your way down, then once you reach the bottom move to the top right and go to the bottom again. To make sure the team names you put into the list are the same as the team names in the dataframe, put the team name you typed into the line "df_team = df_boxavg[df_boxavg['team'] == "team name you typed"]" and if it has an output it's the correct name. To get an idea of what the team name is, go to sports-reference and search up the college basketball team. e.g. search up Duke and you'll find they're called Duke Blue Devils in the dataframe.

Going over what each function does:

string_modify: adds onto the end of the column names. It's used in the code to add _r to the end of the column names. This isn't necessary and it's more an artifact of when I was trying to had _a away team columns and _h home team columns
scoretrip: takes a teams record in string form 'W-L' and returns an integer value of W and L.
dict_from_lists: takes a list of keys and values and returns a dictionary to be added to a dataframe
datasetup: calculates all the boxscore statistics that each team has from the previous games and creates a dataframe where each row has a teams previous statistics and has the points scored in that current game.
finishdatasetup: takes the dataframe created in datasetup and takes the differences in statistics for each game to set up the dataframe that will be fed into the machine learning model.
year_to_df: combines all the previous functions such that all you need to is input the season year that you want and it outputs the final dataframe of that year that can be fed to the model.

train_test_split: splits the data into training and testing data. Inputs are season_data, perct, and cut_off. season_data is a list with as many seasons as you want, so it could be [df_20,df_21,df_23,df_24] if you want to feed 2020-2024 data into the model or it could just be [df_24]
