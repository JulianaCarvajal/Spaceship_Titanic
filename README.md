# Spaceship_Titanic
For the final project in the Python course, it was proposed to develop a Kaggle competition.

## Set Up
### Requirements
1. kaggle API installed through pip assistance
2. kaggle .json downloaded on C:/Users/User/.kaggle folder
3. Pandas
4. Seaborn
5. Matplotlib
6. sklearn
## Dataset
In this competition the task is to predict whether a passenger was transported to an alternate dimension during the Spaceship Titanic's collision with the spacetime anomaly. The dataset has information on each passenger, which is used to make the prediction.

- train.csv: Personal records for about two-thirds (~8700) of the passengers, to be used as training data.

-PassengerId: A unique Id for each passenger. Each Id takes the form gggg_pp where gggg indicates a group the passenger is travelling with and pp is their number within the group. People in a group are often family members, but not always.

-HomePlanet: The planet the passenger departed from, typically their planet of permanent residence.
-CryoSleep: Indicates whether the passenger elected to be put into suspended animation for the duration of the voyage. Passengers in cryosleep are confined to their cabins.

-Cabin: The cabin number where the passenger is staying. Takes the form deck/num/side, where side can be either P for Port or S for Starboard.
Destination - The planet the passenger will be debarking to.

-Age: The age of the passenger.

-VIP: Whether the passenger has paid for special VIP service during the voyage.

-RoomService, FoodCourt, ShoppingMall, Spa, VRDeck: Amount the passenger has billed at each of the Spaceship Titanic's many luxury amenities.

-Name: The first and last names of the passenger.

-Transported: Whether the passenger was transported to another dimension. This is the target, the column you are trying to predict.

- test.csv: Personal records for the remaining one-third (~4300) of the passengers, to be used as test data. Your task is to predict the value of Transported for the passengers in this set.

- sample_submission.csv: A submission file in the correct format.

-PassengerId: Id for each passenger in the test set.

-Transported: The target. For each passenger, predict either True or False.

