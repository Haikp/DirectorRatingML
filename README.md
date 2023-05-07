# DirectorRatingML
This project appplies to the corresponding .csv file within the repo

Determines the rating of a director based of their past work
Uses Linear Regression (OLS), and Gradient Descent

Determined the compatability of the dataset with linear regression
After trial and error, it has been determined that linear regression can be used

Algorithm:
- For fun, testing the compatability of the data set with Linear Regression
  - attempt to salvage certain parts of the data set with missing values
      - from testing, budget has good correlation in determining the gross revenue
      - dependent variable: gross
      - independent varaible: only budget
        - initially, the number of votes was intially believed to be a factor, however upon using visual aids, there were no correlations
- Main algorithm
  - use of gradient descent to allow some tuning capabilities as opposed to linear regression (OLS)
  - determine variables to use
    - rating of movie will represent the directors' rating, thus being the dependent variable
    - of 15 features, only 3 were used, the other data were not ordinal data
        - Votes, Gross, and Budget
        - Votes: refers to number of votes a movie got for their rating
        - Gross and Budget: Combined into total income ratio
          - normalizes values by being concerned with the ratio of profit to budget, rather than the raw amount
            - solves having millions of dollars be next to thousands of dollars
          - also allows us to use a 3D graph, having 2 independent variables and 1 dependent variable
  - with a scale of 0 to 10, we have an RMSE value of .02 - .03, meaning it is fairly accurate in determining ratings
