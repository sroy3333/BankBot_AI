# services/atm_service.py

import random

def get_nearest_atm_distance(location=None, user_distance=None):
    # If user explicitly mentioned distance
    if user_distance:
        return user_distance

    # If location is known, simulate city-based distance
    if location:
        return f"{round(random.uniform(0.3, 2.0), 1)} km"

    # Generic fallback
    return f"{round(random.uniform(0.5, 1.5), 1)} km"

