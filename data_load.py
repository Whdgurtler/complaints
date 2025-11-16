import pandas as pd

def load_data():
    dta = pd.read_csv('../data/complaints.csv')
    return dta
def fetch_complaints(api_url, params, max_records=50):
    all_complaints = []
    total_fetched = 0
    page = 1
    records_per_request = params.get('size', 10)

    while total_fetched < max_records:
        params['page'] = page
        response = requests.get(api_url, params=params)
        data = response.text
        #print(data)
        #write to file for debugging
        with open('debug_response.txt', 'w') as f:
            f.write(data)
        
        
        data = json.loads(data)
        # If data is a dict, get 'results'; if it's a list, use it directly
        if isinstance(data, dict):
            complaints = data.get('results', [])
        elif isinstance(data, list):
            complaints = data
        else:
            complaints = []
        if not complaints:
            break
        all_complaints.extend(complaints)
        total_fetched += len(complaints)
        page += 1
        time.sleep(1)

    return all_complaints[:max_records]
### Example usage
# api_url = 'https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/'
# params = {
#     'size': 10,
#     'company': 'UNITED SERVICES AUTOMOBILE ASSOCIATION',
#     'sort': 'created_date_desc',
#     'format': 'json'
# }
# complaints = fetch_complaints(api_url, params, max_records=5)
# print(f"Fetched {len(complaints)} complaints")
# print(complaints)