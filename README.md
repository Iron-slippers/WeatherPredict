# WeatherPredict
The decision of the hackathon on the topic of weather prediction based on data from the past.

### Setting the task
Participants are asked to algorithmically predict weather conditions based on some measurement history. The teams perform the following tasks:

1) Having a dataset with weather indicators for 43 hours of observations, make a prediction of the same parameters for the next 5 hours.
2) Upload the results to the Telegram bot, get a quality assessment. Teams compete in the accuracy of predictions (you can make an unlimited number of attempts).
3) Presentation of the team's work (publication of code, slides).

### Data
The data includes measurements of various indicators at points (latitude, longitude) on a 30x30 grid with a step of about 5 km (y - from north to south, x - from west to east).

All arrays except heights as the first dimension have a time scale in increments of 1 hour.

### Solution
Checked ideas:
1) common_model - good weather prediction accuracy is shown by the weather today will be the same as yesterday (no use);
2) base_model - create coefficients between yesterday and today for each point on the map. Then we multiply yesterday's evening weather parameters by the coefficients;
3) catboost - features were created for cottbus: the number of the hour in the day, the delta for the last one, two ..., five hours;
4) elastic - use features of catboost (no use);
5) mix_model - combining catboost and base_model. 
