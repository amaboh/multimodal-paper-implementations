import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import datetime, time
import pytz
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def calculate_volatility_estimators(price_df, min_periods=20):
    """
    Calculate the Garman-Klass and Parkinson volatility estimators with error handling.
    
    Args:
        price_df: DataFrame with price data (Open, High, Low, Close)
        min_periods: Minimum number of valid observations for calculation
        
    Returns:
        DataFrame with additional volatility columns
    """
    # Create a copy to avoid modifying the original
    result_df = price_df.copy()
    
    # Ensure required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close']
    missing_columns = [col for col in required_columns if col not in result_df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for negative or zero prices (data errors)
    for col in required_columns:
        if (result_df[col] <= 0).any():
            logger.warning(f"Found non-positive values in {col}. Replacing with NaN.")
            result_df.loc[result_df[col] <= 0, col] = np.nan
    
    try:
        # Parkinson volatility estimator (Equation 12 in the paper)
        # Using log prices to handle large price moves more effectively
        result_df['vol_PK'] = np.sqrt(
            (np.log(result_df['High'] / result_df['Low'])**2) / (4 * np.log(2))
        )
        
        # Garman-Klass volatility estimator (Equation 13 in the paper)
        result_df['vol_GK'] = np.sqrt(
            0.5 * np.log(result_df['High'] / result_df['Low'])**2 -
            (2 * np.log(2) - 1) * np.log(result_df['Close'] / result_df['Open'])**2
        )
        
        # Deal with any invalid results (infinity, NaN)
        result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Remove obvious outliers (more than 5 standard deviations from mean)
        for col in ['vol_PK', 'vol_GK']:
            mean = result_df[col].mean()
            std = result_df[col].std()
            lower_bound = mean - 5 * std
            upper_bound = mean + 5 * std
            result_df.loc[(result_df[col] < lower_bound) | (result_df[col] > upper_bound), col] = np.nan
        
        logger.info(f"Calculated volatility estimators. Found {result_df['vol_GK'].isna().sum()} NaN values.")
    
    except Exception as e:
        logger.error(f"Error calculating volatility estimators: {e}")
        # Add empty columns to ensure they exist
        result_df['vol_PK'] = np.nan
        result_df['vol_GK'] = np.nan
    
    return result_df

def calculate_price_features(price_df, include_additional_features=True):
    """
    Calculate the daily returns features with enhanced error handling and additional features.
    
    Args:
        price_df: DataFrame with price data
        include_additional_features: Whether to include additional features beyond those in the paper
        
    Returns:
        DataFrame with price features
    """
    # Create a copy to avoid modifying the original
    result_df = price_df.copy()
    
    # Ensure required columns exist
    required_columns = ['Open', 'High', 'Low', 'Close']
    missing_columns = [col for col in required_columns if col not in result_df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    try:
        # Calculate previous day's close
        result_df['Prev_Close'] = result_df['Close'].shift(1)
        
        # Calculate returns (Equation 33 in the paper)
        result_df['Open_Return'] = result_df['Open'] / result_df['Prev_Close'] - 1
        result_df['High_Return'] = result_df['High'] / result_df['Prev_Close'] - 1
        result_df['Low_Return'] = result_df['Low'] / result_df['Prev_Close'] - 1
        result_df['Close_Return'] = result_df['Close'] / result_df['Prev_Close'] - 1
        
        # Deal with any invalid results (infinity, NaN)
        result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        if include_additional_features:
            # Add additional features that might be useful
            
            # Trading volume features (if available)
            if 'Volume' in result_df.columns:
                # Volume change
                result_df['Volume_Change'] = result_df['Volume'].pct_change()
                
                # Normalized volume (z-score of log volume over 20-day window)
                log_volume = np.log(result_df['Volume'].replace(0, np.nan))
                result_df['Volume_Z'] = (log_volume - log_volume.rolling(20).mean()) / log_volume.rolling(20).std()
            
            # Price range as a percentage of previous close
            result_df['Range_Pct'] = (result_df['High'] - result_df['Low']) / result_df['Prev_Close']
            
            # Moving averages
            result_df['MA5'] = result_df['Close'].rolling(5).mean() / result_df['Prev_Close'] - 1
            result_df['MA10'] = result_df['Close'].rolling(10).mean() / result_df['Prev_Close'] - 1
            result_df['MA20'] = result_df['Close'].rolling(20).mean() / result_df['Prev_Close'] - 1
            
            # Relative Strength Index (simplified version)
            delta = result_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            result_df['RSI'] = 100 - (100 / (1 + rs))
            
            # Average True Range (ATR) normalized by close price
            tr1 = result_df['High'] - result_df['Low']
            tr2 = abs(result_df['High'] - result_df['Close'].shift(1))
            tr3 = abs(result_df['Low'] - result_df['Close'].shift(1))
            tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
            result_df['ATR'] = tr.rolling(14).mean() / result_df['Close']
        
        # Create price features dataframe
        if include_additional_features:
            price_features = result_df[['Open_Return', 'High_Return', 'Low_Return', 'Close_Return', 
                                       'Range_Pct', 'MA5', 'MA10', 'MA20', 'RSI', 'ATR'] + 
                                      (['Volume_Change', 'Volume_Z'] if 'Volume' in result_df.columns else [])].copy()
        else:
            price_features = result_df[['Open_Return', 'High_Return', 'Low_Return', 'Close_Return']].copy()
        
        # Remove NaN values from the first rows where features couldn't be calculated
        # Determine the maximum window used in calculations
        max_window = 20 if include_additional_features else 1
        price_features = price_features.iloc[max_window:].copy()
        
        logger.info(f"Calculated price features. Shape after processing: {price_features.shape}")
        
    except Exception as e:
        logger.error(f"Error calculating price features: {e}")
        # Return a minimal dataframe with the required columns
        price_features = pd.DataFrame(columns=['Open_Return', 'High_Return', 'Low_Return', 'Close_Return'])
    
    return price_features

def preprocess_headlines(headlines_df, word_embeddings, min_words=3, max_token_length=50):
    """
    Preprocess headlines into word indices for embedding lookup with better error handling.
    
    Args:
        headlines_df: DataFrame with headline data
        word_embeddings: Word embedding model
        min_words: Minimum number of tokens to consider a headline valid
        max_token_length: Maximum length of a token to consider
        
    Returns:
        Processed headlines with tokenized and indexed words
    """
    stop_words = set(stopwords.words('english'))
    # Add stock market specific stopwords
    additional_stops = {'corp', 'inc', 'ltd', 'co', 'corporation', 'company', 'plc', 'group'}
    stop_words.update(additional_stops)
    
    lemmatizer = WordNetLemmatizer()
    
    processed_headlines = []
    skipped_count = 0
    
    for _, row in tqdm(headlines_df.iterrows(), total=len(headlines_df), desc="Processing headlines"):
        try:
            if 'headline' not in row or pd.isna(row['headline']) or not isinstance(row['headline'], str):
                skipped_count += 1
                continue
                
            headline = row['headline']
            
            # Clean text: remove special characters, except ' and -
            headline = re.sub(r'[^a-zA-Z0-9\s\'\-]', ' ', headline)
            # Remove extra whitespace
            headline = re.sub(r'\s+', ' ', headline).strip()
            
            # Tokenize
            tokens = word_tokenize(headline.lower())
            
            # Remove stopwords and lemmatize
            filtered_tokens = []
            for token in tokens:
                # Skip very long tokens (likely errors or concatenated words)
                if len(token) > max_token_length:
                    continue
                    
                # Skip stopwords
                if token.lower() in stop_words:
                    continue
                    
                # Lemmatize and add to filtered tokens
                lemma = lemmatizer.lemmatize(token)
                filtered_tokens.append(lemma)
            
            # Skip headlines with too few tokens
            if len(filtered_tokens) < min_words:
                skipped_count += 1
                continue
            
            # Convert to indices (words in embedding vocabulary)
            indices = []
            for token in filtered_tokens:
                if token in word_embeddings.key_to_index:
                    indices.append(word_embeddings.key_to_index[token])
                # Handle unknown words by skipping them (could also use a special UNK token)
            
            # Only add headlines that have at least some tokens in the embedding vocabulary
            if indices:
                date_str = row['date'] if isinstance(row['date'], str) else row['date'].strftime('%Y-%m-%d')
                time_str = row['time'] if 'time' in row and row['time'] else '12:00:00'  # Default to noon if no time
                
                processed_headlines.append({
                    'date': date_str,
                    'time': time_str,
                    'time_category': row.get('time_category', 'unknown'),
                    'text': headline,
                    'indices': indices,
                    'tokens': filtered_tokens
                })
            else:
                skipped_count += 1
                
        except Exception as e:
            logger.error(f"Error processing headline: {e}")
            skipped_count += 1
    
    logger.info(f"Processed {len(processed_headlines)} headlines, skipped {skipped_count}")
    
    return processed_headlines

def categorize_news_by_market_hours(headlines_df, timezone='US/Eastern'):
    """
    Categorize news into before_market, during_market, after_market categories with timezone handling.
    
    Args:
        headlines_df: DataFrame with headline data including time
        timezone: Timezone string for market hours (default is US/Eastern for NYSE)
        
    Returns:
        DataFrame with additional time_category column
    """
    result_df = headlines_df.copy()
    
    # Add timezone if not already in pytz
    try:
        tz = pytz.timezone(timezone)
    except:
        logger.warning(f"Unknown timezone: {timezone}. Using US/Eastern instead.")
        tz = pytz.timezone('US/Eastern')
    
    # Define market hours (NYSE: 9:30 AM to 4:00 PM Eastern)
    market_open = time(9, 30)
    market_close = time(16, 0)
    
    try:
        # Ensure datetime columns are properly formatted
        if 'datetime' not in result_df.columns:
            # Try to combine date and time columns
            if 'date' in result_df.columns and 'time' in result_df.columns:
                # Convert date to datetime if it's not already
                if not pd.api.types.is_datetime64_any_dtype(result_df['date']):
                    result_df['date'] = pd.to_datetime(result_df['date'], errors='coerce')
                
                # Convert time strings to time objects
                if isinstance(result_df['time'].iloc[0], str):
                    # Parse time strings
                    result_df['time_obj'] = result_df['time'].apply(
                        lambda x: datetime.strptime(x, '%H:%M:%S').time() if isinstance(x, str) else None
                    )
                else:
                    # If time is already a time object
                    result_df['time_obj'] = result_df['time']
                
                # Combine date and time
                result_df['datetime'] = result_df.apply(
                    lambda row: datetime.combine(row['date'].date(), row['time_obj']) 
                                if pd.notna(row['date']) and row['time_obj'] 
                                else None, 
                    axis=1
                )
                
                # Drop temporary column
                result_df.drop('time_obj', axis=1, inplace=True)
            else:
                logger.error("Cannot create datetime: missing date or time columns")
                # Create a placeholder column
                result_df['datetime'] = pd.NaT
        
        # Extract time of day for categorization
        result_df['time_of_day'] = result_df['datetime'].apply(
            lambda x: x.time() if pd.notna(x) else None
        )
        
        # Categorize based on market hours
        def categorize_time(t):
            if pd.isna(t) or t is None:
                return 'unknown'
            if t < market_open:
                return 'before_market'
            elif t > market_close:
                return 'after_market'
            else:
                return 'during_market'
        
        result_df['time_category'] = result_df['time_of_day'].apply(categorize_time)
        
        # Check for weekends and holidays
        result_df['weekday'] = result_df['datetime'].apply(
            lambda x: x.weekday() if pd.notna(x) else None
        )
        
        # Mark weekends as "before_market" for the next trading day
        # (simplified approach; a real implementation would use a trading calendar)
        weekend_mask = result_df['weekday'].isin([5, 6])  # 5=Saturday, 6=Sunday
        result_df.loc[weekend_mask, 'time_category'] = 'before_market'
        
        logger.info(f"Categorized news: {result_df['time_category'].value_counts().to_dict()}")
        
    except Exception as e:
        logger.error(f"Error categorizing news by market hours: {e}")
        # Create a default column if processing fails
        result_df['time_category'] = 'unknown'
    
    return result_df

def align_news_with_prices(headlines_data, price_data):
    """
    Align news headlines with price data by date and categorize news.
    Implements the alignment described in the paper where after-market news
    are grouped with pre-market news of the next trading day.
    
    Args:
        headlines_data: Dictionary of processed headlines by stock
        price_data: Dictionary of price dataframes by stock
        
    Returns:
        Dictionary of aligned news data by stock and date
    """
    aligned_data = {}
    
    for symbol, headlines in headlines_data.items():
        if symbol not in price_data:
            logger.warning(f"No price data available for {symbol}, skipping alignment")
            continue
            
        prices_df = price_data[symbol]
        trading_dates = set(prices_df.index.date)
        
        aligned_headlines = {}
        
        # Initialize empty lists for each trading date
        for date in trading_dates:
            aligned_headlines[date] = []
        
        # Group headlines by date
        for headline in headlines:
            try:
                # Convert string date to datetime date object
                headline_date = datetime.strptime(headline['date'], '%Y-%m-%d').date()
                
                # Handle news alignment according to time category
                # After-market news affect the next trading day
                if headline['time_category'] == 'after_market':
                    # Find the next trading date
                    next_dates = [d for d in trading_dates if d > headline_date]
                    if next_dates:
                        next_trading_date = min(next_dates)
                        if next_trading_date in aligned_headlines:
                            headline['aligned_date'] = next_trading_date
                            aligned_headlines[next_trading_date].append(headline)
                else:
                    # Before-market and during-market news affect the current day
                    if headline_date in trading_dates:
                        headline['aligned_date'] = headline_date
                        aligned_headlines[headline_date].append(headline)
            except Exception as e:
                logger.error(f"Error aligning headline {headline}: {e}")
        
        aligned_data[symbol] = aligned_headlines
        
        # Log some statistics
        total_aligned = sum(len(headlines) for headlines in aligned_headlines.values())
        logger.info(f"{symbol}: Aligned {total_aligned} headlines with {len(trading_dates)} trading days")
    
    return aligned_data

def prepare_sliding_window_samples(aligned_news, price_data, window_size=10):
    """
    Prepare samples using a sliding window approach for both news and price data.
    
    Args:
        aligned_news: Dictionary of aligned news by stock and date
        price_data: Dictionary of price dataframes by stock
        window_size: Size of the sliding window (in days)
        
    Returns:
        List of sample dictionaries with windows of news and price data
    """
    samples = []
    
    for symbol, news_by_date in aligned_news.items():
        if symbol not in price_data:
            continue
            
        prices_df = price_data[symbol]
        
        # Sort dates to ensure chronological order
        dates = sorted(news_by_date.keys())
        
        # Create sliding windows
        for i in range(len(dates) - window_size):
            # Window of dates
            window_dates = dates[i:i+window_size]
            
            # Target date (next day after window)
            if i + window_size < len(dates):
                target_date = dates[i + window_size]
            else:
                continue  # Skip if we don't have a target date
            
            # Check if target volatility is available
            if target_date not in prices_df.index or 'vol_GK' not in prices_df.columns or np.isnan(prices_df.loc[target_date, 'vol_GK']):
                continue
                
            # Get price features for the window
            window_prices = []
            skip_sample = False
            
            for date in window_dates:
                # Convert date to the format used in prices_df index if needed
                if isinstance(prices_df.index[0], pd.Timestamp):
                    lookup_date = pd.Timestamp(date)
                else:
                    lookup_date = date
                
                # Skip if price data is not available for this date
                if lookup_date not in prices_df.index:
                    skip_sample = True
                    break
                
                # Get price features
                if all(feat in prices_df.columns for feat in ['Open_Return', 'High_Return', 'Low_Return', 'Close_Return']):
                    features = prices_df.loc[lookup_date, ['Open_Return', 'High_Return', 'Low_Return', 'Close_Return']].values
                    window_prices.append(features)
                else:
                    skip_sample = True
                    break
            
            if skip_sample or len(window_prices) < window_size:
                continue
            
            # Get headlines for the window
            window_headlines = []
            for date in window_dates:
                day_headlines = news_by_date.get(date, [])
                window_headlines.append(day_headlines)
            
            # Get target volatility
            if isinstance(prices_df.index[0], pd.Timestamp):
                target_lookup = pd.Timestamp(target_date)
            else:
                target_lookup = target_date
                
            target_volatility = prices_df.loc[target_lookup, 'vol_GK']
            
            # Add sample
            samples.append({
                'symbol': symbol,
                'window_dates': window_dates,
                'target_date': target_date,
                'price_features': np.array(window_prices),
                'headlines': window_headlines,
                'target_volatility': target_volatility
            })
    
    logger.info(f"Created {len(samples)} sliding window samples")
    return samples

def create_embedding_matrix(word_to_idx, word_embeddings):
    """
    Create an embedding matrix for the vocabulary.
    
    Args:
        word_to_idx: Dictionary mapping words to indices
        word_embeddings: Word embedding model
        
    Returns:
        Numpy array of embedding vectors
    """
    embedding_dim = word_embeddings.vector_size
    embedding_matrix = np.zeros((len(word_to_idx), embedding_dim))
    
    for word, idx in word_to_idx.items():
        if word in word_embeddings:
            embedding_matrix[idx] = word_embeddings[word]
    
    logger.info(f"Created embedding matrix with shape {embedding_matrix.shape}")
    return embedding_matrix

def split_data_by_time(samples, train_end='2013-12-31', val_end='2015-12-31'):
    """
    Split samples into training, validation, and test sets by date.
    
    Args:
        samples: List of sample dictionaries
        train_end: End date for training data
        val_end: End date for validation data
        
    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    train_end_date = datetime.strptime(train_end, '%Y-%m-%d').date()
    val_end_date = datetime.strptime(val_end, '%Y-%m-%d').date()
    
    train_samples = []
    val_samples = []
    test_samples = []
    
    for sample in samples:
        target_date = sample['target_date']
        
        if target_date <= train_end_date:
            train_samples.append(sample)
        elif target_date <= val_end_date:
            val_samples.append(sample)
        else:
            test_samples.append(sample)
    
    logger.info(f"Split data: {len(train_samples)} train, {len(val_samples)} validation, {len(test_samples)} test samples")
    return train_samples, val_samples, test_samples

def build_vocabulary(headlines_data, min_freq=5):
    """
    Build a vocabulary from processed headlines with minimum frequency threshold.
    
    Args:
        headlines_data: Dictionary of processed headlines by stock
        min_freq: Minimum frequency for a word to be included
        
    Returns:
        Dictionary mapping words to indices
    """
    word_counts = {}
    
    # Count word frequencies
    for symbol, headlines in headlines_data.items():
        for headline in headlines:
            for token in headline.get('tokens', []):
                word_counts[token] = word_counts.get(token, 0) + 1
    
    # Filter by frequency
    filtered_words = [word for word, count in word_counts.items() if count >= min_freq]
    
    # Create word-to-index mapping
    word_to_idx = {word: idx+1 for idx, word in enumerate(filtered_words)}
    word_to_idx['<PAD>'] = 0  # Add padding token
    
    logger.info(f"Built vocabulary with {len(word_to_idx)} words")
    return word_to_idx

def analyze_data_statistics(samples, price_data, headlines_data):
    """
    Analyze and log statistics about the dataset.
    
    Args:
        samples: List of sample dictionaries
        price_data: Dictionary of price dataframes by stock
        headlines_data: Dictionary of headlines by stock
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    # Sample statistics
    stats['num_samples'] = len(samples)
    
    symbols = set(sample['symbol'] for sample in samples)
    stats['num_symbols'] = len(symbols)
    
    # Count samples per symbol
    symbol_counts = {}
    for sample in samples:
        symbol = sample['symbol']
        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
    
    stats['samples_per_symbol'] = symbol_counts
    
    # Headline statistics
    total_headlines = 0
    headlines_per_symbol = {}
    
    for symbol, headlines in headlines_data.items():
        num_headlines = len(headlines)
        total_headlines += num_headlines
        headlines_per_symbol[symbol] = num_headlines
    
    stats['total_headlines'] = total_headlines
    stats['headlines_per_symbol'] = headlines_per_symbol
    stats['avg_headlines_per_symbol'] = total_headlines / len(headlines_data) if headlines_data else 0
    
    # Headline length statistics
    headline_lengths = []
    
    for symbol, headlines in headlines_data.items():
        for headline in headlines:
            tokens = headline.get('tokens', [])
            headline_lengths.append(len(tokens))
    
    if headline_lengths:
        stats['avg_headline_length'] = sum(headline_lengths) / len(headline_lengths)
        stats['min_headline_length'] = min(headline_lengths)
        stats['max_headline_length'] = max(headline_lengths)
        
        # Distribution of headline lengths
        length_dist = {}
        for length in headline_lengths:
            length_dist[length] = length_dist.get(length, 0) + 1
        stats['headline_length_distribution'] = length_dist
    
    # Volatility statistics
    all_volatilities = []
    
    for sample in samples:
        all_volatilities.append(sample['target_volatility'])
    
    if all_volatilities:
        stats['avg_volatility'] = sum(all_volatilities) / len(all_volatilities)
        stats['min_volatility'] = min(all_volatilities)
        stats['max_volatility'] = max(all_volatilities)
        stats['median_volatility'] = sorted(all_volatilities)[len(all_volatilities)//2]
    
    # Log key statistics
    logger.info(f"Dataset statistics:")
    logger.info(f"  Number of samples: {stats['num_samples']}")
    logger.info(f"  Number of symbols: {stats['num_symbols']}")
    logger.info(f"  Total headlines: {stats['total_headlines']}")
    logger.info(f"  Average headlines per symbol: {stats['avg_headlines_per_symbol']:.2f}")
    
    if 'avg_headline_length' in stats:
        logger.info(f"  Average headline length: {stats['avg_headline_length']:.2f} tokens")
    
    if 'avg_volatility' in stats:
        logger.info(f"  Average target volatility: {stats['avg_volatility']:.6f}")
        logger.info(f"  Volatility range: {stats['min_volatility']:.6f} - {stats['max_volatility']:.6f}")
    
    return stats