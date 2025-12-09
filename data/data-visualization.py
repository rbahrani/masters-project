import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("stocks_processed_final.csv")
df_original = pd.read_csv("stocks.csv")

def show_daily_returns(df):

    # remove the outliers
    non_outliers = df[(df["daily_return"] <= 0.5)]
    non_outliers = non_outliers[(non_outliers["daily_return"] >= -0.5)]
    returns = non_outliers["daily_return"] * 100

    # Create the plot with figure size
    plt.figure(figsize=(12, 5))

    # add titles, x and y labels
    plt.title("Occurrences of Different Daily Returns")
    plt.xlabel("Stock Daily Return %")
    plt.ylabel("Occurrences")
    plt.hist(returns, bins=100)
    plt.grid(alpha=0.5)
    # show the plot
    # plt.show()

    # save the plot
    plt.savefig("show_daily_returns.png")

def show_headline_per_stock(df):

    # Count occurrences per ticker
    ticker_counts = df["stock"].value_counts()
    top_10 = ticker_counts.iloc[:10]

    # Create the plot with figure size
    plt.figure(figsize=(12, 7))
    top_10.plot(kind="bar")

    # add titles, x and y labels
    plt.ylabel("Count of Headlines")
    plt.title("Dataset # of News Headlines for Each Stock")
    plt.xlabel("Stock")
    plt.grid(alpha=0.5, axis="y")
    # show the plot
    # plt.show()

    # save the plot
    plt.savefig("show_headline_per_stock.png")


def show_headline_volume_over_time(df):
    # calculate the count of headlines per day
    count_of_headlines_per_date = df.groupby("date").size()

    # print for analysis purposes
    print("Day with max number of headlines: ", count_of_headlines_per_date.idxmax(), "")

    # create plot with figure size
    plt.figure(figsize=(10, 5))
    count_of_headlines_per_date.plot(kind="line")

    # add titles, x and y labels
    plt.title("Dataset's # of Headlines per Day")
    plt.xlabel("Date")
    plt.ylabel("Count of Headlines")
    plt.grid(alpha=0.3)
    # show the plot
    # plt.show()

    # save the plot
    plt.savefig("show_headline_volume_over_time.png")

def show_distribution_of_headline_length(df):
    # calculate the length of each headline
    str_lengths = df["headline"].str.len()

    # create plot with figure size
    plt.figure(figsize=(10, 7))
    sns.histplot(str_lengths, bins=100)

    # add titles, x and y labels
    plt.title("Distribution of Lengths of Headline Strings")
    plt.xlabel("Length of Headline Strings")
    plt.ylabel("Occurrences")

    # show the plot
    # plt.show()

    # save the plot
    plt.savefig("show_distribution_of_headline_length.png")


def get_statistical_analysis_on_raw_data(df):
    print("======================")
    # Analysis of the news headlines
    df['headline_length'] = df['headline'].str.len()
    print(df['headline_length'].describe())
    print("======================")

    # Median, Min and Max headline lengths
    print("Median headline length is: ", df['headline_length'].median())
    print("Max headline length is: ", df['headline_length'].max())
    print("Min headline length is: ", df['headline_length'].min())
    print("======================")

def get_statistical_analysis_on_processed_data(df):
    print("======================")
    print("Analysis of daily returns:")
    daily_returns = df["daily_return"]
    print(daily_returns.describe())
    print(daily_returns.info())
    print("======================")

    # Analyzing the min and max daily returns
    print("Max daily return is: ", daily_returns.max())
    print("Min daily return is: ", daily_returns.min())
    print("Median daily return is: ", daily_returns.median())
    print("======================")

    print("Analysis of stock open prices:")
    # Analysing the open prices of the stocks
    open_prices = df["open_price"]
    print(open_prices.describe())
    print(open_prices.info())
    print("======================")

    # Analysing min, max, and median of the open prices
    print("Max open price is: ", open_prices.max())
    print("Min open price is: ", open_prices.min())
    print("Median open price is: ", open_prices.median())
    print("======================")

    print("Analysis of stock close prices:")
    # Analysing the close prices of the stocks
    close_prices = df["close_price"]
    print(close_prices.describe())
    print(close_prices.info())
    print("======================")

    # Analysing min, max, and median of the close prices
    print("Max close price is: ", close_prices.max())
    print("Min close price is: ", close_prices.min())
    print("Median close price is: ", close_prices.median())
    print("======================")


if __name__ == "__main__":
    show_daily_returns(df)
    show_headline_per_stock(df)
    show_headline_volume_over_time(df)
    show_distribution_of_headline_length(df_original)
    get_statistical_analysis_on_raw_data(df_original)
    get_statistical_analysis_on_processed_data(df)