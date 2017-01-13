### Ad Sniper

This program was developed as part of a group competition to create an application that makes use of cell phone data analytics. The data set consisted of hundreds of millions of data point describing app-usage data from users on a certain mobile carrier. By using k-means clustering, my team discovered that there were clusters of users that conformed to the same app-usage patterns; for example, users that favored apps categorized as 'Education' also tended to favor 'Trivia' apps, but showed little preference for 'Sports' apps. We used these discovered trends to develop a method for intelligent ad targeting for companies that wish to deploy a specific number of ad campaigns. For instance, a company might have the budget to deliver five social media ad campaigns and want to ensure that each campaign targets users in a meaningful way. In addition to the experience gained with k-means, this project was also an exercise in data engineering; in order to make use of the raw data, it had to be inspected, reduced, and cleaned, a task made more challenging by the massive size of the data set.

### Gallery

![img](https://github.com/tphinkle/AdSniper/blob/master/data/plots/sports_game_users_behavior.png?raw=true)

This plot shows a clear correlation between 'sports' and 'action' app users, a relationship that could be used to target both types of app users with similar ads.


![img](https://github.com/tphinkle/AdSniper/blob/master/data/plots/corr_heatmap.png?raw=true)

Correlation matrix of all app categories.

![img](https://github.com/tphinkle/AdSniper/blob/master/data/plots/4_1.png?raw=true)

Projection of the app-usage data and k-means centroids (5-means) on two app categories.



