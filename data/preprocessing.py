import pandas as pd
import yfinance as yf
import time
from yfinance.exceptions import YFRateLimitError

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i + n]

def print_most_common_occurance(data):
    # print the top 100 most common tickers (just for visualization purposes)
    print(data["stock"].value_counts().head(100))

def remove_rows_with_missing_values(data):
    # drop any rows with missing values
    return data.dropna()

def preprocess_headline_strings(data):
    # ensure the headline is a string
    data["headline"] = data["headline"].astype(str)
    # convert all headlines to lowercase characters
    data["headline"] = data["headline"].str.lower()
    # remove all non-alphanumeric characters
    data["headline"] = data["headline"].str.replace(r"[^0-9a-zA-Z ]+", " ", regex=True)
    # remove extra whitespace
    data["headline"] = data["headline"].str.replace(r"\s+", " ", regex=True).str.strip()
    return data

def fix_date_formats(data):
    # update all date format to "%Y-%m-%d"
    return data.assign(date=lambda df: pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d"))

def download_prices_slow_run(tickers, start, end, batch_size=5, max_retries=6, base_sleep_between_batches=2, base_sleep_on_rate_limit=60):
    all_batches = []
    i = 0
    for batch in chunked(tickers, batch_size):
        # loop through each batch of tickers
        print(f"\nBatch {i+1}: {len(batch)} tickers {batch}")
        i += 1
        for attempt in range(max_retries):
            # for each batch, do max_tries of attempts to get the data
            try:
                # print attempt number
                print("Attempt: ", attempt+1)

                # most important: download data from yfinance API
                current_res = yf.download(batch, start=start, end=end, group_by="ticker", auto_adjust=False, progress=False, threads=False)

                # if data is empty, print message
                if current_res is None or current_res.empty:
                    print("Empty batch result")
                else:
                    # if data is not empty, append to list and print success message
                    all_batches.append(current_res)
                    print(f"Success. Sleeping {base_sleep_between_batches} seconds")
                    time.sleep(base_sleep_between_batches)
                    break  # next batch
            except Exception as e:
                # if an exception occurs, check if it's a rate limiting error from YFinance
                is_rate_limited = isinstance(e, YFRateLimitError) or "Rate limited" in str(e)
                if is_rate_limited:
                    # if rate limiting error, sleep and try again on the next attempt
                    print(f"Rate limited. Sleeping {base_sleep_on_rate_limit} seconds")
                    time.sleep(base_sleep_on_rate_limit)
                    continue
                else:
                    # if another error, skip the batch
                    print("Different error. Other issues could be wrong, skipping this batch", e)
                    break # next batch
        else:
            print("Failed all attempts. Skipping batch id: ", i)

    # return the concatenated DataFrame
    price_data = pd.concat(all_batches, axis=1)
    return price_data

def augment_stocks_open_close_prices(data):

    # make a copy to keep the original df safe
    df = data.copy()

    # Normalize dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()

    # get the list of unique tickers that need to be passed into yfinance API for price retrieval
    unique_tickers = df["stock"].dropna().unique().tolist()

    # calculate the range of dates that we need data from
    min_date = df["date"].min()
    max_date = df["date"].max()
    print("min_date:", min_date, "max_date:", max_date)

    # pass in the tickers and min and max dates to download prices from yfinance API
    price_data = download_prices_slow_run(tickers=unique_tickers, start=min_date, end=max_date + pd.Timedelta(days=1))

    # keep only Open and Close columns
    price_oc = price_data.loc[:, (slice(None), ["Open", "Close"])]

    # Stack tickers into rows
    stacked = price_oc.stack(level=0).reset_index()

    # adjust the naming for the first two columns if they are wrong
    date_raw_col = stacked.columns[0]
    ticker_raw_col = stacked.columns[1]

    stacked.rename(columns={date_raw_col: "date", ticker_raw_col: "stock", "Open": "open_price", "Close": "close_price"}, inplace=True)

    # Normalize dates
    stacked["date"] = pd.to_datetime(stacked["date"]).dt.normalize()

    # only keep the columns we care about and do a left join with the original df
    stacked = stacked[["stock", "date", "open_price", "close_price"]]
    merged = df.merge(stacked, on=["stock", "date"], how="left")

    # print length of rows for sanity check
    print("Merged rows:", len(merged))
    return merged

def calculate_daily_returns(data):
    # calculate daily returns: (close price - open price) / open price
    # which also equals to close price / open price - 1
    df = data.assign(daily_return=lambda df: df["close_price"] / df["open_price"] - 1)
    return df.dropna(subset=["daily_return"])

def save_df_to_csv(df, filename):
    df.to_csv(filename ,index=False)

if __name__ == "__main__":
    stocks_df = pd.read_csv('C:/Users/rosie/dtsc-691/data/stocks.csv', encoding='utf-8')
    # print_most_common_occurance(stocks_df)
    stocks_df = remove_rows_with_missing_values(stocks_df)
    stocks_df = preprocess_headline_strings(stocks_df)
    stocks_df = fix_date_formats(stocks_df)
    stocks_df = augment_stocks_open_close_prices(stocks_df)
    stocks_df = calculate_daily_returns(stocks_df)


    save_df_to_csv(stocks_df, 'C:/Users/rosie/dtsc-691/data/stocks_processed_final.csv')
