import requests
import os
import io

import pandas as pd

from argparse import ArgumentParser
from time import sleep
from tqdm import tqdm
from typing import Optional


def download_data(symbol, postfix='.us') -> Optional[pd.DataFrame]:
    url = f'https://stooq.pl/q/d/l/?s={symbol}{postfix}&i=d'
    data = requests.get(url).text
    if data == 'Brak danych':
        print(f'No data found for {symbol}{postfix}')
        return None
    if 'Przekroczony dzienny limit wywolan' in data:
        print(f'Request limit reached for today!!! - {data}')
        raise RuntimeError('Out of requests')
    return pd.read_csv(io.StringIO(data))


def save_history(df: pd.DataFrame, index, symbol):
    if not os.path.exists(f'data/{index}/history'):
        os.mkdir(f'data/{index}/history')
    df.to_csv(f'data/{index}/history/{symbol}.csv', index=False)


def main(index, limit, sleep_time):
    companies = pd.read_csv(f'data/{index}/companies.csv')
    index_data = download_data(index, postfix='')

    if index_data is not None:
        print('Successfully downloaded index history')
        save_history(index_data, index, index)

    summaries = []
    downloaded_companies = 0

    if limit is not None:
        companies = companies.iloc[:limit]

    for company in tqdm(companies['Symbol']):
        if os.path.exists(f'data/{index}/history/{company}.csv'):
            continue

        data = download_data(company, postfix="")

        summaries.append({
            'symbol': company,
            'rows': len(data.index) if data is not None else pd.NA
        })

        if data is not None:
            save_history(data, index, company)
            downloaded_companies += 1

        sleep(sleep_time)

    print(f'Founded {downloaded_companies} companies histories')
    summaries_df = pd.DataFrame(summaries)
    summaries_df.to_csv(f'data/{index}/summary.csv', index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-s', '--sleep', help='Sleep time between each GET', type=float, default=1)
    parser.add_argument('-i', '--index', help='Requested index', type=str, required=True)
    parser.add_argument('-l', '--limit', help='Limit number of companies to first N', type=int)

    args = parser.parse_args()

    main(args.index, args.limit, args.sleep)
