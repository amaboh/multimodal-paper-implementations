import pandas as pd
import numpy as np
import yfinance as yf
import os
import requests
import time
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import json
from tqdm import tqdm

def download_stock_prices(stock_symbols, start_date='2007-01-01', end_date='2017-12-31', cache_dir='data/prices'):
    """
    Download daily stock price data for the given stock symbols with caching mechanism.
    
    Args:
        stock_symbols: List of stock ticker symbols
        start_date: Start date for data collection
        end_date: End date for data collection
        cache_dir: Directory to cache downloaded data
        
    Returns:
        Dictionary of pandas DataFrames with stock price data
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    price_data = {}
    
    for symbol in tqdm(stock_symbols, desc="Downloading stock price data"):
        cache_file = os.path.join(cache_dir, f"{symbol}.csv")
        
        # Check if cached data exists
        if os.path.exists(cache_file):
            try:
                stock_data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                print(f"Loaded cached data for {symbol} ({len(stock_data)} days)")
                price_data[symbol] = stock_data
                continue
            except Exception as e:
                print(f"Error loading cached data for {symbol}: {e}. Will download fresh data.")
        
        try:
            # Use yfinance to download the data
            stock_data = yf.download(symbol, start=start_date, end=end_date)
            
            if not stock_data.empty:
                price_data[symbol] = stock_data
                
                # Cache the data
                stock_data.to_csv(cache_file)
                print(f"Downloaded {len(stock_data)} days of data for {symbol}")
            else:
                print(f"No data found for {symbol}")
        except Exception as e:
            print(f"Error downloading {symbol}: {e}")
            
        # Add a small delay to avoid hitting API rate limits
        time.sleep(0.5)
    
    return price_data

def collect_reuters_headlines(stock_symbols, company_names, start_date='2007-01-01', end_date='2017-12-31', cache_dir='data/headlines'):
    """
    Alternative approach to collect financial news headlines using a free financial news API.
    This implementation uses Alpha Vantage API as an example, but can be adapted for other sources.
    
    Args:
        stock_symbols: List of stock ticker symbols
        company_names: Dictionary mapping stock symbols to company names and aliases
        start_date: Start date for data collection
        end_date: End date for data collection
        cache_dir: Directory to cache downloaded headlines
        
    Returns:
        Dictionary of dataframes with headlines for each stock
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    headlines_data = {}
    
    # For a real implementation, you would use an API key from Alpha Vantage, News API, or similar
    # API_KEY = "YOUR_API_KEY"  # Replace with your actual API key
    
    # Since we don't have an actual API key in this example, let's create a function
    # that simulates API calls and returns sample data based on the stock symbol
    def simulate_news_api(symbol, company_aliases, from_date, to_date):
        """Simulate API response for testing purposes"""
        # In a real implementation, you would use:
        # url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={API_KEY}"
        # response = requests.get(url)
        # data = response.json()
        
        # For simulation, create sample data
        sample_data = []
        
        # Convert dates to datetime objects
        start_dt = datetime.strptime(from_date, '%Y-%m-%d')
        end_dt = datetime.strptime(to_date, '%Y-%m-%d')
        
        # Generate some sample headlines for each month in the range
        current_dt = start_dt
        while current_dt <= end_dt:
            # Generate more headlines during earnings seasons (Q1, Q2, Q3, Q4)
            is_earnings_month = current_dt.month in [1, 4, 7, 10]
            num_headlines = np.random.randint(2, 10) if is_earnings_month else np.random.randint(0, 5)
            
            for _ in range(num_headlines):
                # Randomize the day within the month
                day = np.random.randint(1, 28)
                news_date = datetime(current_dt.year, current_dt.month, day)
                
                # Skip if before start_date or after end_date
                if news_date < start_dt or news_date > end_dt:
                    continue
                
                # Randomize the hour to distribute across market hours
                hour = np.random.randint(0, 24)
                minute = np.random.randint(0, 60)
                
                # Create a timestamp
                timestamp = news_date.replace(hour=hour, minute=minute)
                date_str = timestamp.strftime('%Y-%m-%d')
                time_str = timestamp.strftime('%H:%M:%S')
                
                # Choose a sample headline based on company and randomize
                company_name = company_aliases[0]  # Use first alias
                
                # Create more realistic headlines based on patterns
                headline_templates = [
                    f"{company_name} reports Q{(current_dt.month-1)//3+1} earnings {['above', 'below', 'meeting'][np.random.randint(0,3)]} expectations",
                    f"{company_name} announces new {['product line', 'partnership', 'expansion', 'CEO', 'strategy'][np.random.randint(0,5)]}",
                    f"{company_name} shares {['rise', 'fall', 'remain stable'][np.random.randint(0,3)]} after {['earnings report', 'analyst upgrade', 'market volatility', 'sector news'][np.random.randint(0,4)]}",
                    f"Analysts {['upgrade', 'downgrade', 'maintain'][np.random.randint(0,3)]} {company_name} stock",
                    f"{company_name} {['cuts', 'raises', 'maintains'][np.random.randint(0,3)]} {['dividend', 'forecast', 'guidance'][np.random.randint(0,3)]}",
                    f"{company_name} to {['acquire', 'merge with', 'partner with'][np.random.randint(0,3)]} {['competitor', 'startup', 'tech company'][np.random.randint(0,3)]}",
                    f"{company_name} {['beats', 'misses', 'meets'][np.random.randint(0,3)]} on {['revenue', 'profit', 'growth targets'][np.random.randint(0,3)]}"
                ]
                
                headline = headline_templates[np.random.randint(0, len(headline_templates))]
                
                sample_data.append({
                    'date': date_str,
                    'time': time_str,
                    'headline': headline
                })
            
            # Move to next month
            if current_dt.month == 12:
                current_dt = datetime(current_dt.year + 1, 1, 1)
            else:
                current_dt = datetime(current_dt.year, current_dt.month + 1, 1)
        
        return sample_data
    
    for symbol in tqdm(stock_symbols, desc="Collecting headlines"):
        cache_file = os.path.join(cache_dir, f"{symbol}_headlines.csv")
        
        # Check if cached headlines exist
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, parse_dates=['date'])
                headlines_data[symbol] = df
                print(f"Loaded cached headlines for {symbol}: {len(df)} headlines")
                continue
            except Exception as e:
                print(f"Error loading cached headlines for {symbol}: {e}. Will collect new data.")
        
        # Get company aliases
        aliases = company_names.get(symbol, [symbol])
        
        try:
            # In a real implementation, you would call your news API here
            # For this example, we're using a simulation function
            headlines = simulate_news_api(symbol, aliases, start_date, end_date)
            
            if headlines:
                # Convert to DataFrame
                df = pd.DataFrame(headlines)
                
                # Cache the data
                df.to_csv(cache_file, index=False)
                
                headlines_data[symbol] = df
                print(f"Collected {len(df)} headlines for {symbol}")
        except Exception as e:
            print(f"Error collecting headlines for {symbol}: {e}")
        
        # Add a delay to avoid hitting API rate limits
        time.sleep(0.5)
    
    return headlines_data

def compile_company_surface_forms(stock_symbols):
    """
    Create a dictionary of company names and their surface forms using a more robust approach.
    
    Args:
        stock_symbols: List of stock symbols
        
    Returns:
        Dictionary mapping stock symbols to lists of surface forms
    """
    # Define some common surface forms for major companies
    known_forms = {
        'AAPL': ['Apple', 'Apple Inc', 'Apple Computer'],
        'MSFT': ['Microsoft', 'Microsoft Corp', 'Microsoft Corporation'],
        'AMZN': ['Amazon', 'Amazon.com', 'Amazon Inc'],
        'GOOGL': ['Google', 'Alphabet', 'Alphabet Inc', 'Google LLC'],
        'FB': ['Facebook', 'Meta', 'Meta Platforms', 'Facebook Inc'],
        'TSLA': ['Tesla', 'Tesla Inc', 'Tesla Motors'],
        'PG': ['Procter & Gamble', 'Procter and Gamble', 'P&G'],
        'WMT': ['Walmart', 'Wal-Mart', 'Wal Mart', 'Walmart Inc'],
        'JPM': ['JPMorgan Chase', 'JPMorgan', 'JP Morgan', 'J.P. Morgan'],
        'XOM': ['Exxon Mobil', 'ExxonMobil', 'Exxon'],
        'JNJ': ['Johnson & Johnson', 'Johnson and Johnson', 'J&J'],
        'KO': ['Coca-Cola', 'Coca Cola', 'Coke', 'The Coca-Cola Company'],
        'PEP': ['PepsiCo', 'Pepsi', 'Pepsi Co'],
        'COST': ['Costco', 'Costco Wholesale', 'Costco Wholesale Corporation'],
        'CVS': ['CVS Health', 'CVS', 'CVS Pharmacy'],
        'MO': ['Altria', 'Altria Group', 'Philip Morris'],
        'WBA': ['Walgreens', 'Walgreens Boots Alliance', 'Walgreen'],
        'MDLZ': ['Mondelez', 'Mondelez International', 'Kraft'],
        'CL': ['Colgate-Palmolive', 'Colgate', 'Palmolive'],
        'CVX': ['Chevron', 'Chevron Corporation', 'Standard Oil of California'],
        'COP': ['ConocoPhillips', 'Conoco', 'Phillips'],
        'EOG': ['EOG Resources', 'EOG', 'Enron Oil & Gas'],
        'OXY': ['Occidental Petroleum', 'Occidental', 'Oxy'],
        'VLO': ['Valero Energy', 'Valero', 'Valero Energy Corporation'],
        'HAL': ['Halliburton', 'Halliburton Company'],
        'SLB': ['Schlumberger', 'Schlumberger Limited'],
        'PXD': ['Pioneer Natural Resources', 'Pioneer'],
        'APC': ['Anadarko Petroleum', 'Anadarko'],
        'NEE': ['NextEra Energy', 'NextEra', 'Florida Power & Light'],
        'DUK': ['Duke Energy', 'Duke', 'Duke Power'],
        'SO': ['Southern Company', 'Southern Co', 'Southern'],
        'D': ['Dominion Energy', 'Dominion', 'Dominion Resources'],
        'EXC': ['Exelon', 'Exelon Corporation'],
        'AEP': ['American Electric Power', 'AEP'],
        'SRE': ['Sempra Energy', 'Sempra'],
        'PEG': ['Public Service Enterprise Group', 'PSEG', 'PSE&G'],
        'ED': ['Consolidated Edison', 'Con Edison', 'ConEd'],
        'XEL': ['Xcel Energy', 'Xcel', 'Northern States Power'],
        'UNH': ['UnitedHealth Group', 'UnitedHealth', 'United Healthcare'],
        'PFE': ['Pfizer', 'Pfizer Inc'],
        'MRK': ['Merck', 'Merck & Co', 'Merck and Co'],
        'MDT': ['Medtronic', 'Medtronic plc'],
        'AMGN': ['Amgen', 'Amgen Inc'],
        'ABT': ['Abbott Laboratories', 'Abbott', 'Abbott Labs'],
        'GILD': ['Gilead Sciences', 'Gilead', 'GILD'],
        'LLY': ['Eli Lilly', 'Lilly', 'Eli Lilly and Company'],
        'BMY': ['Bristol-Myers Squibb', 'BMS', 'Bristol Myers', 'Bristol-Myers'],
        'BAC': ['Bank of America', 'BofA', 'Bank of America Corporation'],
        'WFC': ['Wells Fargo', 'Wells Fargo & Company'],
        'C': ['Citigroup', 'Citi', 'Citibank', 'Citicorp'],
        'GS': ['Goldman Sachs', 'Goldman', 'Goldman Sachs Group'],
        'USB': ['U.S. Bancorp', 'US Bancorp', 'US Bank'],
        'MS': ['Morgan Stanley', 'MS'],
        'AXP': ['American Express', 'AmEx', 'Amex'],
        'PNC': ['PNC Financial Services', 'PNC Bank', 'PNC']
    }
    
    surface_forms = {}
    
    for symbol in stock_symbols:
        # Use known forms if available
        if symbol in known_forms:
            surface_forms[symbol] = known_forms[symbol]
        else:
            # If not available, try to get company name from yfinance (fallback)
            try:
                ticker = yf.Ticker(symbol)
                company_name = ticker.info.get('shortName', symbol)
                surface_forms[symbol] = [company_name, symbol]
            except:
                # If all else fails, just use the symbol itself
                surface_forms[symbol] = [symbol]
    
    return surface_forms

# Function to download company data from finviz as an additional data source
def download_company_data_from_finviz(stock_symbols, cache_dir='data/company_info'):
    """
    Scrape basic company information from Finviz for each stock symbol.
    
    Args:
        stock_symbols: List of stock ticker symbols
        cache_dir: Directory to cache downloaded data
        
    Returns:
        Dictionary of company information
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    company_data = {}
    
    for symbol in tqdm(stock_symbols, desc="Downloading company data"):
        cache_file = os.path.join(cache_dir, f"{symbol}_info.json")
        
        # Check if cached data exists
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    company_data[symbol] = json.load(f)
                print(f"Loaded cached company data for {symbol}")
                continue
            except Exception as e:
                print(f"Error loading cached company data for {symbol}: {e}. Will download fresh data.")
        
        try:
            # In a real implementation, you would scrape Finviz here
            # For this example, we'll use a simulation function
            company_info = simulate_company_info(symbol)
            
            # Cache the data
            with open(cache_file, 'w') as f:
                json.dump(company_info, f)
            
            company_data[symbol] = company_info
            print(f"Downloaded company data for {symbol}")
        except Exception as e:
            print(f"Error downloading company data for {symbol}: {e}")
        
        # Add a small delay to avoid hitting website rate limits
        time.sleep(1)
    
    return company_data

def simulate_company_info(symbol):
    """Simulate company information for testing purposes"""
    # In a real implementation, you would scrape this data
    sectors = {
        'PG': 'Consumer Staples', 'KO': 'Consumer Staples', 'PEP': 'Consumer Staples',
        'WMT': 'Consumer Staples', 'COST': 'Consumer Staples', 'CVS': 'Consumer Staples',
        'MO': 'Consumer Staples', 'WBA': 'Consumer Staples', 'MDLZ': 'Consumer Staples',
        'CL': 'Consumer Staples',
        
        'XOM': 'Energy', 'CVX': 'Energy', 'COP': 'Energy', 'EOG': 'Energy',
        'OXY': 'Energy', 'VLO': 'Energy', 'HAL': 'Energy', 'SLB': 'Energy',
        'PXD': 'Energy', 'APC': 'Energy',
        
        'NEE': 'Utilities', 'DUK': 'Utilities', 'SO': 'Utilities', 'D': 'Utilities',
        'EXC': 'Utilities', 'AEP': 'Utilities', 'SRE': 'Utilities', 'PEG': 'Utilities',
        'ED': 'Utilities', 'XEL': 'Utilities',
        
        'JNJ': 'Healthcare', 'UNH': 'Healthcare', 'PFE': 'Healthcare', 'MRK': 'Healthcare',
        'MDT': 'Healthcare', 'AMGN': 'Healthcare', 'ABT': 'Healthcare', 'GILD': 'Healthcare',
        'LLY': 'Healthcare', 'BMY': 'Healthcare',
        
        'BRK-A': 'Financials', 'JPM': 'Financials', 'BAC': 'Financials', 'WFC': 'Financials',
        'C': 'Financials', 'GS': 'Financials', 'USB': 'Financials', 'MS': 'Financials',
        'AXP': 'Financials', 'PNC': 'Financials'
    }
    
    # Return some basic information
    return {
        'symbol': symbol,
        'sector': sectors.get(symbol, 'Unknown'),
        'company_name': compile_company_surface_forms([symbol])[symbol][0],
        'market_cap': f"${np.random.randint(1, 1000)} B" if symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN'] else f"${np.random.randint(1, 100)} B"
    }