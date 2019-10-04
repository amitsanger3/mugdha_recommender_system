# Import some useful lib
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#import statistics as st
#import sqlite3 as sq
from sklearn.model_selection import train_test_split
#import datetime
#import math


class DishPram(object):
    def __init__(self, dish_id, dish, pram):
        """
        dish_id: list or dataset of dish's id.
        dish: list or dataset of dish's.
        pram: list or dataset of parameter on which
            we get our recommendations like: mean,
            co-occurrence etc.
        """
        self.dish_id = dish_id
        self.dish = dish
        self.pram = pram

    def get_dish_id(self):
        return self.dish_id

    def get_dish(self):
        return self.dish

    def get_pram(self):
        return self.pram

    def __str__(self):
        return str(self.dish_id)+" " + self.dish + ": " +str(self.pram)



class Recommendations(object):

    def __init__(self):
        self.dish_id = None
        self.dishes = None
        self.pram = None
        self.matrix_mean = None
        self.guest_id = None


    def get_dish_id(self):
        return self.dish_id

    def get_dishes(self):
        return self.dish


    def get_pram(self):
        """
        Recently used parameters list for reccomendations
        """
        return self.pram


    def get_recent_matrix_mean(self):
        """
        Recently calculated mean of a matrix
        """
        return self.matrix_mean


    def get_guest_id(self):
        return self.guest_id


    def get_recommendation_by_pram(self, dish_id, dish, pram):
        """
        get_recommendation_by_pram(self, dish_id, dish, pram)

        Gives first five recommendations as per the descending
        order of the given params.

        Parameters:
        dish_id= List of id'd of dishes.
        dish= List of dishes.
        pram= List of parameters like mean or
            co-occurence etc.

        *NOTE: dish_id, dish & pram all 3 are of same sizes.
        """
        self.dish_id = dish_id
        self.dish = dish
        self.pram = pram

        dish_list = [] # For carrying dish details
        for i in range(len(self.dish_id)):
            dish_list.append(DishPram(dish_id=self.dish_id[i], dish=self.dish[i], pram=self.pram[i]))

        recommendation = sorted(dish_list, key=Recommendations.get_pram, reverse=True)
        #[print(i) for i in recommendation[:11]]

        return recommendation[:11]


    def get_training_test_data(self, df_order, tr, ts, r):
        """
        get_training_test_data(self, df_order=pandas.dataframe, tr=float(0,1), ts=float(0,1), r=int)
        Getting randomly select training & testing data.

        Parameters:
        df_order = pandas.dataframe of guests order.
        ts = float of % of df_order need for training size.
        tr = float of % of df_order need for test size.
        r = random.seed
        """
        df_order_train, df_order_test = train_test_split(df_order, test_size=ts, train_size=tr, random_state=r)

        return df_order_train, df_order_test


    def get_order_and_ratings_matrix(self, orders):
        """
        Matrix of guest's orders of size(no. of total guests attended,no. of all
        dish served), of values 0 & 1.
        1: If guest placed an order of dish then(guest_id_index, dish_id_index)=1
        0: otherwise

        Matrix of ratings given by guests to the perticular dishes. If a guest gives
        multiple time ratings to a dish, it takes mean of all ratings.
        0.0 otherwise.

        Parameters:
        orders= Dataframe of orders query of guets, carrying guest id of particular
                guest, dish ids of the dishes he rated & the ratings he gave to the
                particular dish.
            """
        guest_rating_matrix=orders.pivot_table(values='guest_order_ratings',
                                                index='guest_to_attend_id',
                                                columns='guest_order_id',
                                                aggfunc='mean',
                                                fill_value=0,
                                                margins=False,
                                                dropna=True,
                                                margins_name='All',)

        guest_order_matrix = (guest_rating_matrix>0).astype(dtype=int)

        return guest_order_matrix, guest_rating_matrix


    def get_matrix_mean(self, matrix):
        """
        get_matrix_mean(self, ratings_matrix=matrix)
        List of column-wise means of a matrix

        Parameter:
        matrix= Matrix of which means to be calculated.
        """
        matrix_mean = (matrix.mean(axis=0)).to_list()
        self.matrix_mean = matrix_mean
        return matrix_mean


    def get_co_occurrence_matrix(self, order_matrix):
        """
        Co-occurence matrix of orders. If any combination of dishes match
        with order adds 1 with the value, otherwise 0.

        Parameter:
        order_matrix= matrix of guests orders.
        """
        order_matrix_trans = np.transpose(order_matrix) #Transpose of order matrix
        co_occurrence_matrix = np.array(order_matrix_trans.dot(order_matrix)) #dot product
        np.fill_diagonal(co_occurrence_matrix, 0) #setting 0 values at the diagonal

        # getting sum of co_occurrence matrix column wise & reshape it in n*1 matrix
        co_occurrence_matrix_sum = co_occurrence_matrix.sum(axis=0).reshape(len(co_occurrence_matrix),1)

        # deviding co_occurence matrix row wise to co_occurence_matrix_sum
        co_occurrence_matrix_mean=pd.DataFrame(np.array(co_occurrence_matrix)*(1/co_occurrence_matrix_sum))

        return co_occurrence_matrix_mean


    def get_recommendations_by_highest_ratings(self, dish_id, dish, rating_matrix):
        """
        get_recommendations_by_highest_ratings(self, dish_id=list, dish=list, rating_matrix=matrix)

        Top 5 recommendations of highest ratings

        Parameters:
        dish_id = list of all dish'ids
        dish = list of all the dishes
        rating_matrix = matrix of rating's means of the ratings given by guests to
                        the dishes they ordered.
        """
        rating_matrix_mean = self.get_matrix_mean(rating_matrix)
        recommendations_by_highest_ratings = self.get_recommendation_by_pram(dish_id, dish, rating_matrix_mean)

        return recommendations_by_highest_ratings


    def get_recommendations_by_popularity(self, dish_id, dish, order_matrix):
        """
        get_recommendations_by_popularity(self, dish_id=list, dish=list, order_matrix=matrix)

        Top 5 recommendations of highest selling dishes

        Parameters:
        dish_id = list of all dish'ids
        dish = list of all the dishes
        order_matrix = matrix of 1 or 0 if guest orders that perticular dish or not,
                        respectively.
        """
        order_matrix_mean = self.get_matrix_mean(order_matrix)
        recommendations_by_popularity = self.get_recommendation_by_pram(dish_id, dish, order_matrix_mean)

        return recommendations_by_popularity


    def get_recommendations_by_occurence(self, guest_tried_dishes, dish_id, dishes, co_occurrence_matrix):
        """
        get_recommendations_by_occurence(self, guest_tried_dishes=list, dish_id=list, dishes=list, co_occurrence_matrix=matrix)

        Get 5 recommendations on the basis of nearest neighbour method.

        Parameters:
        guest_tried_dishes = list id dishes's ids that a single perticuler
                                user aleady tried. eg. [dish23, dish47] only
                                two if he orders these two in his earlier
                                visits.
        dish_id = list of all dish'ids
        dish = list of all the dishes
        co_occurrence_matrix = matrix of integers ranges(0, len(dish_id)),
                                depends number of times any combination of
                                dishes get ordered.
        """
        co_occurrence_list = []
        for i in guest_tried_dishes:
            co_occurrence_data = co_occurrence_matrix.iloc[dish_id.index(i)]
            co_occurrence_list.append(co_occurrence_data)

        df_mean = pd.DataFrame(co_occurrence_list).mean()

        recommendations_by_occurence = self.get_recommendation_by_pram(dish_id, dishes, df_mean)

        return recommendations_by_occurence
