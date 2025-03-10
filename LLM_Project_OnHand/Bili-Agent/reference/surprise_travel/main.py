#!/usr/bin/env python
import sys
from crew import SurpriseTravelCrew
import json
# from crewai.crews.crew_output import CrewOutput


def run():
    # Replace with your inputs, it will automatically interpolate any tasks and agents information
    inputs = {
        'origin': 'Hongkong, HKG',
        'destination': 'Xian, XIY',
        'age': 20,
        'hotel_location': 'Xian',
        'flight_information': 'GOL 1234, leaving at Dec 20th, 2024, 10:00',
        'trip_duration': '14 days'
    }
    result = SurpriseTravelCrew().crew().kickoff(inputs=inputs)
    print(result.to_dict())
    print(type(result.to_dict()))
    res = result.to_dict()

    with open('output.json', 'w') as json_file:
        json.dump(res, json_file)


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        'origin': 'São Paulo, GRU',
        'destination': 'New York, JFK',
        'age': 31,
        'hotel_location': 'Brooklyn',
        'flight_information': 'GOL 1234, leaving at June 30th, 2024, 10:00',
        'trip_duration': '14 days'
    }
    try:
        SurpriseTravelCrew().crew().train(n_iterations=int(sys.argv[1]), inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")


if __name__ == '__main__':
    run()
    # res = {'name': "Exciting Xi'an Exploration", 'day_plans': [{'date': 'Day 1', 'activities': [{'name': 'Explore the Terracotta Army Museum', 'location': 'Terracotta Army Museum', 'description': 'Discover the ancient army of life-sized terracotta sculptures', 'date': 'Day 1', 'cousine': 'Local gourmet cuisine', 'why_its_suitable': "Experience the rich history and culture of Xi'an", 'reviews': None, 'rating': None}, {'name': 'Visit the Big Wild Goose Pagoda', 'location': 'Big Wild Goose Pagoda', 'description': 'Explore the ancient Buddhist pagoda and its surroundings', 'date': 'Day 1', 'cousine': 'Local gourmet cuisine', 'why_its_suitable': "Enjoy the iconic landmark of Xi'an", 'reviews': None, 'rating': None}], 'restaurants': ["Chang'an Hui"], 'flight': 'Departure from [Origin] to Xian Airport (XIY)'}, {'date': 'Day 2', 'activities': [{'name': 'Morning food tour in the Muslim Quarter', 'location': 'Muslim Quarter', 'description': 'Embark on a culinary adventure through the bustling streets', 'date': 'Day 2', 'cousine': 'Local delicacies', 'why_its_suitable': "Experience the vibrant food culture of Xi'an", 'reviews': None, 'rating': None}, {'name': 'Leisurely bike ride on the Ancient City Wall', 'location': 'Ancient City Wall', 'description': 'Enjoy a scenic bike ride atop the ancient city wall', 'date': 'Day 2', 'cousine': 'Unique dining experience', 'why_its_suitable': "Admire the panoramic views of Xi'an", 'reviews': None, 'rating': None}, {'name': "Stroll through the Great Mosque of Xi'an", 'location': "Great Mosque of Xi'an", 'description': 'Immerse yourself in the traditional architecture and peaceful ambiance', 'date': 'Day 2', 'cousine': 'Unique dining experience', 'why_its_suitable': "Experience the cultural heritage of Xi'an", 'reviews': None, 'rating': None}], 'restaurants': ["Zuì Cháng'ān"], 'flight': None}, {'date': 'Day 3', 'activities': [{'name': 'Huashan Mountain day tour for hiking adventure', 'location': 'Huashan Mountain', 'description': 'Embark on a thrilling hiking expedition in the scenic mountains', 'date': 'Day 3', 'cousine': 'Authentic Chinese dishes', 'why_its_suitable': 'Explore the natural beauty and challenge yourself', 'reviews': None, 'rating': None}, {'name': 'Explore the stunning landscapes of Huashan Mountain', 'location': 'Huashan Mountain', 'description': 'Marvel at the breathtaking scenery and picturesque views', 'date': 'Day 3', 'cousine': 'Authentic Chinese dishes', 'why_its_suitable': 'Connect with nature and unwind in tranquility', 'reviews': None, 'rating': None}], 'restaurants': ['Sānyuán Lǎo Huáng Jiā'], 'flight': None}, {'date': 'Day 4', 'activities': [{'name': 'Discover the history of the Shaanxi History Museum', 'location': 'Shaanxi History Museum', 'description': 'Immerse yourself in the ancient artifacts and cultural relics', 'date': 'Day 4', 'cousine': 'Refreshing dining experience', 'why_its_suitable': 'Learn about the rich history of the region', 'reviews': None, 'rating': None}, {'name': 'Wander through the Small Wild Goose Pagoda', 'location': 'Small Wild Goose Pagoda', 'description': 'Explore the ancient pagoda and its tranquil surroundings', 'date': 'Day 4', 'cousine': 'Refreshing dining experience', 'why_its_suitable': 'Experience serenity amidst historical sites', 'reviews': None, 'rating': None}], 'restaurants': ['Lotus Restaurant'], 'flight': None}], 'hotel': 'Check-in at [Hotel Name], located in [Hotel Address]'}
