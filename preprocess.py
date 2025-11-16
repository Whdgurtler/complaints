import time
import pandas as pd
default to one year ago
from datetime import date, timedelta
def preprocess_data(first_date_to_keep = date.today() - timedelta(days=365), companies_to_keep: list = None):
    
    if companies_to_keep is None:
        companies_to_keep = ['CAPITAL ONE FINANCIAL CORPORATION',
                          'JPMORGAN CHASE & CO.',
                          "Block, Inc.",
                          "WELLS FARGO & COMPANY",
                          "BANK OF AMERICA, NATIONAL ASSOCIATION",
                          "CITIBANK, N.A.",
                          "Early Warning Services, LLC",
                          "NAVY FEDERAL CREDIT UNION",
                          "SYNCHRONY FINANCIAL",
                          "AMERICAN EXPRESS COMPANY",
                          "DISCOVER BANK",
                          "Paypal Holdings, Inc",
                          "U.S. BANCORP",
                          "Chime Financial Inc",
                          "ALLY FINANCIAL INC."
                          ]


    from data_load import load_data
    dta = load_data()
    
    #lowercase column names and replace spaces with underscores
    dta.columns = dta.columns.str.lower().str.replace(' ', '_')
    
    #remove rows with missing consumer complaint narrative
    dta = dta.dropna(subset=['Consumer complaint narrative'])

    #convert date received to datetime
    dta['date_received'] = pd.to_datetime(dta['date_received'])
    #convert date_recieved to date only
    dta['date_received'] = dta['date_received'].dt.date

    #filter to only include rows with date_received on or after first_date_to_keep
    dta = dta[dta['date_received'] >= first_date_to_keep]

    dta['narr_length'] = dta.consumer_complaint_narrative.str.len()
    #drop narrative gt 4930 characters
    dta = dta.loc[dta.narr_length <= 4930, :]
   

    #reset index
    dta = dta.reset_index(drop=True)

    dta = dta.reset_index(drop=True)
    return dta