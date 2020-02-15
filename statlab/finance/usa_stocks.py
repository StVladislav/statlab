import requests
import pandas as pd


"""
This module in progress... .
"""


URL = "https://financialmodelingprep.com/api/v3/"


def fetch_data_from_url(url: str):

    try:
        with requests.get(url) as response:
            data = response.json()
    except Exception:
        raise ValueError

    return data


def fetch_data_from_json(json) -> pd.DataFrame:
    return pd.DataFrame(json)


def get_income_statements(symbol):
    url = URL + f"financials/income-statement/{symbol}?period=quarter"

    return fetch_data_from_url(url)


def get_balance_sheet(symbol):
    url = URL + f"financials/balance-sheet-statement/{symbol}?period=quarter"

    return fetch_data_from_url(url)


def get_cash_flow(symbol):
    url = URL + f"financials/cash-flow-statement/{symbol}?period=quarter"

    return fetch_data_from_url(url)


def get_enterprise_value(symbol):
    url = URL + f"enterprise-value/dis?period=quarter"

    return fetch_data_from_url(url)


def gainer(url):
    pass


def lossers():
    pass


if __name__ == '__main__':
    pass
