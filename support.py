import pandas as pd
import re
def preprocess_date_string(date_string):
    # Regex to match different date formats
    # This handles formats with day, month, year and time components
    match = re.match(r'(\d{1,2})/(\d{1,2})/(\d{2,4}) (\d{1,2}:\d{2})(?:\s?(AM|PM))?', date_string)
    if not match:
        return date_string  # Return original if it doesn't match the expected format
    
    day, month, year, time, am_pm = match.groups()
    # print(day, month, year, time, am_pm)
    # Add leading zeros to day and month if necessary
    day = day.zfill(2)
    month = month.zfill(2)
    # Handle year
    if len(year) == 2:
        # Convert two-digit year to four-digit year
        year = '20' + year

    # if time>12:
        
        
    # Combine parts into a standardized format
    if am_pm:
        date_string = f"{day}/{month}/{year} {time} {am_pm}"
    else:
        date_string = f"{day}/{month}/{year} {time}"

    return date_string

def convert_date(date_string):
    try:
        # Preprocess the date string to ensure consistent formatting
        standardized_date_string = preprocess_date_string(date_string)
#        print(standardized_date_string)
        # Convert date-time string to datetime object
        return pd.to_datetime(standardized_date_string, format='%d/%m/%Y %I:%M %p', errors='coerce')
    except ValueError as e:
        # Print error message and date string for debugging
        print(f"Error parsing date: '{date_string}' -> {e}")
        return pd.NaT  # Return Not a Time for errors