from dateutil.rrule import *
from datetime import datetime
import random
import holidays

# sources:
# https://dateutil.readthedocs.io/en/stable/rrule.html
# https://stackoverflow.com/questions/9755538/how-do-i-create-a-list-of-random-numbers-without-duplicates
# https://stackoverflow.com/questions/2394235/detecting-a-us-holiday

def get_n_random_dates(start_date: datetime, end_date: datetime, n: int, all_dates: bool) -> list[datetime]:
    if (start_date.year % 4 == 0):
        count = 366
    else:
        count = 365
    dates = list(rrule(freq=DAILY, count=count, dtstart=start_date, byweekday=(MO, TU, WE, TH, FR), until=end_date))
    if not all_dates:
        random_indices = random.sample(range(len(dates)), n)
        random_dates = [dates[i] for i in random_indices]
    else:
        random_dates = dates

    # Replace us holidays with new random date
    for date in random_dates:
        if is_us_holiday(date):
            random_dates.remove(date)
            if not all_dates:
                random_index = random.sample(range(len(dates)), 1)[0]
                while random_index in random_indices:
                    random_index = random.sample(range(len(dates)), 1)[0]
                random_dates.append(dates[random_index])
    return random_dates

def month_number_to_name(month_num: int) -> str:
    months_dict = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May",
                   6: "June", 7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
    return months_dict[month_num]

def is_us_holiday(date: datetime) -> bool:
    return (date.date() or date) in holidays.US(years = date.year).keys()

def get_us_holidays(year: int) -> list[datetime]:
    return [date for date in holidays.US(years = year).keys()]
