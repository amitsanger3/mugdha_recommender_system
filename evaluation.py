# Import aome useful library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3 as sq
from sklearn.model_selection import train_test_split
import datetime
import random

# Import Recommendations class
from recommendations import *


class PrecisionRecallEvaluation(object):
    """
    Evaluate Precision-Recall curve on diffrent recommendatons
    """
    
    def __init__(self, test_data, train_data, dish_id, dishes, order_matrix, rating_matrix, co_occurence_matrix):
        """
        Initialize:
        test_data = Part of data we select for testing our algorithm.
        train_data = Part of data we select for train our algorithm.
        dish_id = list of ids of dishes.
        dishes = list of all dishes.
        order_matrix = Matrix of guest's orders of size(no. of total
        guests attended,no. of all dish served), of values 0 & 1.
        rating_matrix = Matrix of ratings given by guests to the 
        perticular dishes. If a guest gives multiple time ratings to
        a dish, it takes mean of all ratings. 0.0 otherwise.
        co_occurence_matrix = Co-occurence matrix of orders. If any
        combination of dishes match with order adds 1 with the value,
        otherwise 0.
        """
        self.test = test_data
        self.train = train_data
        self.dish_id = dish_id
        self.dish = dishes
        self.om = order_matrix
        self.rm = rating_matrix
        self.com = co_occurence_matrix
        
        
    def get_sample(self, population, percent):
        """
        Get percent of sample of given Population.
        
        Parameters;
        population= list of populations samples
        percent = float of percent we need from population
        """
        sample_size = int(len(population)*percent)
        random.seed(0)
        sample = random.sample(population, sample_size)
        
        return sample
    
    
    def commen_guests(self, percent):
        """
        Commen guests from training & testing data set.
        
        Parameters:
        percent=float of percent of commen guests we need.
        """
        guests = list(set(self.train['guest_to_attend_id']).intersection(set(self.test['guest_to_attend_id'])))
        guests_samples = self.get_sample(guests, percent)
        
        return guests_samples
    
    
    def precision_recall_lists(self, percent):
        """
        To get average precision recall lists on diffrent recommendations.
        We iterate on number of recommendations to get that they are in 
        our test set or not. For e.g. first we check of first recomendation
        then first two and then first three and so on.
        
        Parameters;
        percent=float of percent of commen guests we need.
        """
        # Initiakize instance
        rec = Recommendations()
        
        guests = self.commen_guests(percent) # get commen guets
        
        # Dish's id list of Recommendations by popularity
        rec_by_popularity = [i.get_dish_id() for i in rec.get_recommendations_by_popularity(self.dish_id, self.dish, self.om)]
        # Dish's id list of Recommendations by RATINGS
        rec_by_ratings = [i.get_dish_id() for i in rec.get_recommendations_by_highest_ratings(self.dish_id, self.dish, self.rm)]
        
        cut_off = list(range(1,11)) # To iterate number of recommendation in each iteration
        
        rec_hr_avg_precision = [] # Average precision list of recommendations by highest ratings
        rec_hr_avg_recall = [] # Average recall list of recommendations by highest ratings
        
        rec_pop_avg_precision = [] # Average precision list of recommendations by popularity
        rec_pop_avg_recall = [] # Average recall list of recommendations by popularity
        
        rec_com_avg_precision = [] # Average precision list of recommendations by co-occurence
        rec_com_avg_recall = [] # Average recall list of recommendations by co-occurence
        
        for i in cut_off:
            hr_precision = 0.0
            hr_recall = 0.0

            pop_precision = 0.0
            pop_recall = 0.0

            com_precision = 0.0
            com_recall = 0.0
            for guest in guests:
                # guest_set: list of dishes ids which already ordered by guest
                guest_set = list(self.test[self.test['guest_to_attend_id']==int(guest)]['guest_order_id'])     
                # Calculating recommendations by co_occurence
                train_order_dishes = list(self.train[self.train['guest_to_attend_id']==guest]['guest_order_id'])
                rec_by_co_occurence = [i.get_dish_id() for i in rec.get_recommendations_by_occurence(train_order_dishes, self.dish_id, self.dish, self.com)][:i]
                
                # First i recommendations
                rec_by_co_occurence_i = rec_by_co_occurence[:i]
                rec_by_popularity_i = rec_by_popularity[:i]
                rec_by_ratings_i = rec_by_ratings[:i]
                
                # hit_set: Commen dishes in test_set of guest and our recommendations
                hit_set_hr = len(set(guest_set).intersection(set(rec_by_ratings_i))) # hit_set: highest ratings
                hit_set_pop = len(set(guest_set).intersection(set(rec_by_popularity_i))) # hit_set: popularity
                hit_set_com = len(set(guest_set).intersection(set(rec_by_co_occurence_i))) # hit_set: co-occurence
                
                hr_precision += hit_set_hr/float(i)
                hr_recall += hit_set_hr/len(guest_set)
                
                pop_precision += hit_set_pop/float(i)
                pop_recall += hit_set_pop/len(guest_set)
                
                com_precision += hit_set_com/float(i)
                com_recall += hit_set_com/len(guest_set)
                
            rec_hr_avg_precision.append(hr_precision/len(guests))
            rec_hr_avg_recall.append(hr_recall/len(guests))

            rec_pop_avg_precision.append(pop_precision/len(guests))
            rec_pop_avg_recall.append(pop_recall/len(guests))

            rec_com_avg_precision.append(com_precision/len(guests))
            rec_com_avg_recall.append(com_recall/len(guests))
            
        return rec_hr_avg_precision, rec_hr_avg_recall, rec_pop_avg_precision, rec_pop_avg_recall, rec_com_avg_precision, rec_com_avg_recall
    
    
    def precision_recall_evaluation(self, percent):
        """
        Graphical Evaluations of precision-recall curve
        on diffrent Recommendation Algorithm.
        
        Parameters:
        percent=float of percent of commen guests we need.
        
        Output:
        A graphical figure plotting precision-recall curvers
        on diffrent recommendations algorithms.
        """
        hr_precision, hr_recall, pop_precision, pop_recall, com_precision, com_recall = self.precision_recall_lists(percent)
        print('Plotting Precision-Recall curve on diffrent Recommendations \n')
        plt.plot(hr_precision, hr_recall, label='Highest Ratings')
        plt.plot(pop_precision, pop_recall, label='Popularity')
        plt.plot(com_precision, com_recall, label='co-occurence')
        plt.legend()
        plt.title('Evalution on Recommendations')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()
        
        return None

