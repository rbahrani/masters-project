from flask import Flask, request, render_template
import yfinance
from huggingface_hub import hf_hub_download
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from .scraper import fetch_recent_news_for_ticker
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io, base64


class FinBERTRegressor(nn.Module):
    def __init__(self, base_model_name="ProsusAI/finbert", dropout=0.1):
        super().__init__()
        self.base_model = AutoModel.from_pretrained(base_model_name)

        # freeze all the weights in the backbone model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # same regssor head as the best trained model
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.base_model.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, input_ids, attention_mask):

        # run the inputs through the backbone model
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        # get the cls embedding from the output, which is usually index 0
        cls = outputs.last_hidden_state[:, 0, :]

        # run the cls embedding through the regression head
        out = self.regressor(cls)

        # remove the last dimension from the output so the shape is (batch_size,) and return
        return out.squeeze(-1)

# Declare Flask app
app = Flask(__name__)

# import tokenizer from backbone model
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
# update device from cpu/gpu
current_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define the model architecture
model = FinBERTRegressor()

# download the saved model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="rosiebahrani/finbert_regressor_best_revised_mlp",
    filename="finbert_regressor_best_revised_mlp.pt",
    repo_type="model",
)

# load the model weights into the architecture defined above
state_dict = torch.load(model_path, map_location=current_device)
model.load_state_dict(state_dict)

# move the model to the current device
model.to(current_device)

# set the model to evaluation mode
model.eval()

def predict_returns_for_list_of_headlines(headlines, max_length=64):

    # tokenize the headlines
    encodings = tokenizer(headlines, max_length=max_length, padding=True, return_tensors="pt", truncation=True)

    # move the input ids and attention masks to the device
    input_ids = encodings["input_ids"].to(current_device)
    attention_mask = encodings["attention_mask"].to(current_device)

    with torch.no_grad():
        # make predictions for each of the headlines
        predictions = model(input_ids=input_ids, attention_mask=attention_mask)

    # convert the model output from a pytorch tensor to a numpy array
    predictions = predictions.detach().cpu().numpy()

    return predictions

def compute_stats(predictions):

    # calculate the average of the returns
    avg_return = float(np.mean(predictions))

    # calculate the 5th, 50th, and 95th quantiles of the returns
    q05, q50, q95 = np.quantile(predictions, [0.05, 0.5, 0.95])

    # return the statistical summary in a dictionary format
    return {
        "avg": avg_return,
        "q05": float(q05),
        "q50": float(q50),
        "q95": float(q95),
    }

def make_distribution_plot(predictions):
    fig, ax = plt.subplots(figsize=(12, 6))

    # create a plot of the predicted returns
    ax.hist(predictions, bins=300, alpha=0.7)

    # create a grid for the plot for better visibility
    ax.grid(True, alpha=0.3)

    # add x and y labels and titles to the plot
    ax.set_xlabel("Predicted Returns")
    ax.set_ylabel("Occurrences")
    ax.set_title("Distribution of Predicted Returns")

    ### Save the image to a buffer###

    # create an in-memory buffer
    buf = io.BytesIO()
    # adjust the paddings in the plot
    fig.tight_layout()
    # save the plot as a png
    fig.savefig(buf, format="png")
    # move the cursor to the start of the buffer
    buf.seek(0)
    # update the image to a Base64 string
    image_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    # close the plot
    plt.close(fig)

    # return the url for the image
    return f"data:image/png;base64,{image_base64}"

@app.route('/', methods=['GET', 'POST'])
def render_home_page_and_results():

    # provide list of famous tickers for users. This list can get longer in the future to support more stocks
    list_of_tickers = ['AAPL', 'GOOG', 'TSLA', 'AMZN', 'META', 'NVDA', 'AMD', 'JPM', 'SNPS']

    # get the ticker selected by the user
    selected_ticker = request.form.get('selected_ticker')

    # if it's a post request, process the ticker
    if request.method == 'POST':

        # get the data for this ticker from yfinance for the last month
        data = yfinance.Ticker(selected_ticker).history(period='1mo')

        # capture the close prices
        close_prices = data['Close']

        ### create a plot of the close prices ###

        # adjust the background color for the plot
        plt.style.use('dark_background')

        # adjust the size of the plot
        fig, ax = plt.subplots(figsize=(15, 10))

        # plot the close prices with close prices as the y axis and date as the x axis
        ax.plot(data.index, close_prices, linewidth=3)

        # add a title and axis labels for x and y axes
        ax.set_title(f'{selected_ticker} – Last 30 Days (Close Prices)', fontsize=12)
        ax.set_xlabel('Dates', fontsize=8)
        ax.set_ylabel('Prices (USD $)', fontsize=15)

        ###  save the plot to in-memory buffer ###

        # create an in-memory buffer
        buf = io.BytesIO()

        # save as png and trim whitespace around the plot
        fig.savefig(buf, format='png', bbox_inches='tight')

        # move the read/write cursor to the start of the buffer
        buf.seek(0)

        # close the plot
        plt.close(fig)

        # convert the buffer image to a base64 string
        img_64_string = base64.b64encode(buf.getvalue()).decode('utf-8')
        plot_url_30_days = f"data:image/png;base64,{img_64_string}"

        # get the list of relevant articles from web and extract headlines
        articles = fetch_recent_news_for_ticker(selected_ticker)
        headlines = [a["title"] for a in articles]

        # get predictions for the list of headlines
        predictions = predict_returns_for_list_of_headlines(headlines)

        # compute statistical summary on the predictions
        stats = compute_stats(predictions)

        # get a plot for the distribution of predictions
        plot_url = make_distribution_plot(predictions)

        # pass all variables in the template
        return render_template("results.html", selected_ticker=selected_ticker, plot_url_30_days=plot_url_30_days, plot_url=plot_url, articles=articles, predictions=predictions, avg_return=stats["avg"], q05=stats["q05"], q50=stats["q50"], q95=stats["q95"])

    # if it is a GET request, render the dtsc-691.html template
    return render_template("dtsc-691.html", list_of_tickers=list_of_tickers)

@app.route('/resume', methods=['GET'])
def render_resume():
    return render_template("resume.html")

@app.route('/biographical_homepage', methods=['GET'])
def render_biographical_homepage():
    return render_template("biographical_homepage.html")
@app.route('/other_projects', methods=['GET'])
def render_other_projects():
    return render_template("other_projects.html")


if __name__ == "__main__":
    # Start the app
    # LOCAL:
    app.run(host="localhost", port=5000)
    # app.run(debug=True)
