import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt
import os
import pickle
import logging
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import warnings

# Set up logging
logger = logging.getLogger(__name__)

# Filter arch package warnings
warnings.filterwarnings("ignore", message="The scale and shape parameters are redundant", category=UserWarning)
warnings.filterwarnings("ignore", message="Mean effects are not removed in GJR estimation", category=UserWarning)

class GARCHForecaster:
    def __init__(self, p=1, q=1, cache_dir='models/garch', use_cache=True):
        """
        GARCH model forecaster for volatility prediction.
        
        Args:
            p: GARCH lag order
            q: ARCH lag order
            cache_dir: Directory to cache fitted models
            use_cache: Whether to use cached models
        """
        self.p = p
        self.q = q
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.models = {}
        
        if use_cache:
            os.makedirs(cache_dir, exist_ok=True)
    
    def _get_cache_path(self, symbol):
        """Get the cache file path for a symbol"""
        return os.path.join(self.cache_dir, f"garch_{self.p}_{self.q}_{symbol}.pkl")
    
    def fit(self, price_data, symbols=None, max_workers=4):
        """
        Fit GARCH models for the given symbols.
        
        Args:
            price_data: Dictionary of price DataFrames by symbol
            symbols: List of symbols to fit models for (default: all in price_data)
            max_workers: Maximum number of parallel workers
            
        Returns:
            Self (for method chaining)
        """
        if symbols is None:
            symbols = list(price_data.keys())
        
        logger.info(f"Fitting GARCH({self.p},{self.q}) models for {len(symbols)} symbols")
        
        # Define function to fit a single symbol (for parallel processing)
        def fit_single(symbol):
            try:
                cache_path = self._get_cache_path(symbol)
                
                # Try to load from cache
                if self.use_cache and os.path.exists(cache_path):
                    try:
                        with open(cache_path, 'rb') as f:
                            model_result = pickle.load(f)
                            return symbol, model_result
                    except Exception as e:
                        logger.warning(f"Error loading cached model for {symbol}: {e}")
                
                # Get price data
                df = price_data[symbol]
                
                # Calculate returns (r_t = ln(P_t/P_{t-1}))
                if 'log_return' not in df.columns:
                    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
                
                # Remove NaN values
                returns = df['log_return'].dropna()
                
                if len(returns) < 60:  # Need sufficient data to fit model
                    logger.warning(f"Insufficient data for {symbol} ({len(returns)} points), skipping")
                    return symbol, None
                
                # Fit GARCH(p,q) model
                model = arch_model(
                    returns * 100,  # Scale returns to improve numerical stability
                    vol='Garch', 
                    p=self.p, 
                    q=self.q,
                    mean='Zero',  # Use zero mean as in the paper
                    rescale=False
                )
                
                # Fit with robustness options
                result = model.fit(
                    disp='off',
                    show_warning=False,
                    options={'maxiter': 1000}
                )
                
                # Cache the result
                if self.use_cache:
                    try:
                        with open(cache_path, 'wb') as f:
                            pickle.dump(result, f)
                    except Exception as e:
                        logger.warning(f"Error caching model for {symbol}: {e}")
                
                return symbol, result
                
            except Exception as e:
                logger.error(f"Error fitting GARCH model for {symbol}: {e}")
                return symbol, None
        
        # Fit models for all symbols (parallel if multiple workers)
        if max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(tqdm(
                    executor.map(fit_single, symbols),
                    total=len(symbols),
                    desc="Fitting GARCH models"
                ))
        else:
            results = [fit_single(symbol) for symbol in tqdm(symbols, desc="Fitting GARCH models")]
        
        # Store results
        for symbol, result in results:
            if result is not None:
                self.models[symbol] = result
        
        logger.info(f"Successfully fitted {len(self.models)}/{len(symbols)} GARCH models")
        return self
    
    def predict_volatility(self, symbol, last_day, horizon=1):
        """
        Predict volatility for a symbol using the fitted GARCH model.
        
        Args:
            symbol: Stock symbol
            last_day: Last day of observed data
            horizon: Forecast horizon (default: 1 day ahead)
            
        Returns:
            Predicted volatility (or None if model not available)
        """
        if symbol not in self.models:
            logger.warning(f"No fitted model available for {symbol}")
            return None
        
        try:
            model_result = self.models[symbol]
            
            # Get one-step ahead forecast (Equation 5 in the paper)
            forecast = model_result.forecast(horizon=horizon)
            
            # Extract the volatility forecast (convert back from percentage to decimal)
            # Note: GARCH returns variance, so we take the square root to get volatility
            volatility = np.sqrt(forecast.variance.iloc[-1, 0]) / 100
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error predicting volatility for {symbol}: {e}")
            return None
    
    def evaluate(self, price_data, start_date=None, end_date=None):
        """
        Evaluate GARCH models on test data.
        
        Args:
            price_data: Dictionary of price DataFrames by symbol
            start_date: Start date for evaluation period
            end_date: End date for evaluation period
            
        Returns:
            DataFrame with evaluation results
        """
        results = []
        
        for symbol, model_result in tqdm(self.models.items(), desc="Evaluating GARCH models"):
            if symbol not in price_data:
                continue
                
            df = price_data[symbol]
            
            # Filter by date range if specified
            if start_date is not None or end_date is not None:
                date_mask = True
                
                if start_date is not None:
                    date_mask = date_mask & (df.index >= pd.Timestamp(start_date))
                    
                if end_date is not None:
                    date_mask = date_mask & (df.index <= pd.Timestamp(end_date))
                    
                test_df = df[date_mask].copy()
            else:
                test_df = df.copy()
            
            # Skip if no data
            if test_df.empty:
                continue
                
            # Make sure volatility is calculated
            if 'vol_GK' not in test_df.columns or 'vol_PK' not in test_df.columns:
                logger.warning(f"Volatility measures not found for {symbol}, skipping evaluation")
                continue
            
            # Evaluate for each day in the test period
            predictions = []
            
            # Skip the first day as we need previous data for prediction
            for i in range(1, len(test_df)):
                current_date = test_df.index[i-1]
                target_date = test_df.index[i]
                
                # Predict volatility for the next day
                pred_vol = self.predict_volatility(symbol, current_date)
                
                if pred_vol is not None:
                    actual_vol_gk = test_df['vol_GK'].iloc[i]
                    actual_vol_pk = test_df['vol_PK'].iloc[i]
                    
                    predictions.append({
                        'symbol': symbol,
                        'date': target_date,
                        'pred_vol': pred_vol,
                        'actual_vol_gk': actual_vol_gk,
                        'actual_vol_pk': actual_vol_pk,
                        'squared_error_gk': (pred_vol - actual_vol_gk) ** 2,
                        'abs_error_gk': abs(pred_vol - actual_vol_gk),
                        'squared_error_pk': (pred_vol - actual_vol_pk) ** 2,
                        'abs_error_pk': abs(pred_vol - actual_vol_pk)
                    })
            
            if predictions:
                results.extend(predictions)
        
        # Convert to DataFrame
        if not results:
            logger.warning("No evaluation results available")
            return pd.DataFrame()
            
        result_df = pd.DataFrame(results)
        
        # Calculate overall metrics
        metrics = {
            'GK': {
                'MSE': result_df['squared_error_gk'].mean(),
                'MAE': result_df['abs_error_gk'].mean(),
                'R²': self._calculate_r_squared(result_df['pred_vol'], result_df['actual_vol_gk'])
            },
            'PK': {
                'MSE': result_df['squared_error_pk'].mean(),
                'MAE': result_df['abs_error_pk'].mean(),
                'R²': self._calculate_r_squared(result_df['pred_vol'], result_df['actual_vol_pk'])
            }
        }
        
        logger.info(f"GARCH evaluation metrics:")
        logger.info(f"  GK estimator - MSE: {metrics['GK']['MSE']:.6f}, MAE: {metrics['GK']['MAE']:.6f}, R²: {metrics['GK']['R²']:.4f}")
        logger.info(f"  PK estimator - MSE: {metrics['PK']['MSE']:.6f}, MAE: {metrics['PK']['MAE']:.6f}, R²: {metrics['PK']['R²']:.4f}")
        
        return result_df, metrics
    
    def _calculate_r_squared(self, predictions, targets):
        """Calculate coefficient of determination (R²)"""
        if len(predictions) == 0 or len(targets) == 0:
            return 0
            
        # Equation 11 in the paper
        mean_target = targets.mean()
        ss_total = ((targets - mean_target) ** 2).sum()
        ss_residual = ((targets - predictions) ** 2).sum()
        
        if ss_total == 0:
            return 0
            
        r_squared = 1 - (ss_residual / ss_total)
        
        # R² can be negative if the model is worse than the mean
        return max(0, r_squared)
    
    def calculate_sector_metrics(self, evaluation_results, symbol_to_sector):
        """
        Calculate metrics by sector.
        
        Args:
            evaluation_results: DataFrame with evaluation results
            symbol_to_sector: Dictionary mapping symbols to sectors
            
        Returns:
            DataFrame with sector-level metrics
        """
        if evaluation_results.empty:
            return pd.DataFrame()
            
        # Add sector column
        evaluation_results['sector'] = evaluation_results['symbol'].map(
            lambda x: symbol_to_sector.get(x, 'Unknown')
        )
        
        # Group by sector and calculate metrics
        sector_metrics = []
        
        for sector, group in evaluation_results.groupby('sector'):
            # Calculate metrics for Garman-Klass estimator
            gk_mse = group['squared_error_gk'].mean()
            gk_mae = group['abs_error_gk'].mean()
            gk_r2 = self._calculate_r_squared(group['pred_vol'], group['actual_vol_gk'])
            
            # Calculate metrics for Parkinson estimator
            pk_mse = group['squared_error_pk'].mean()
            pk_mae = group['abs_error_pk'].mean()
            pk_r2 = self._calculate_r_squared(group['pred_vol'], group['actual_vol_pk'])
            
            sector_metrics.append({
                'sector': sector,
                'num_predictions': len(group),
                'num_symbols': group['symbol'].nunique(),
                'GK_MSE': gk_mse,
                'GK_MAE': gk_mae,
                'GK_R²': gk_r2,
                'PK_MSE': pk_mse,
                'PK_MAE': pk_mae,
                'PK_R²': pk_r2
            })
        
        return pd.DataFrame(sector_metrics)
    
    def plot_predictions(self, evaluation_results, symbol=None, estimator='GK', 
                        start_date=None, end_date=None, save_path=None):
        """
        Plot predicted vs actual volatility.
        
        Args:
            evaluation_results: DataFrame with evaluation results
            symbol: Symbol to plot (if None, plot aggregate results)
            estimator: Volatility estimator to use ('GK' or 'PK')
            start_date: Start date for plot
            end_date: End date for plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if evaluation_results.empty:
            logger.warning("No evaluation results available for plotting")
            return None
            
        # Filter by symbol if specified
        if symbol is not None:
            df = evaluation_results[evaluation_results['symbol'] == symbol].copy()
            if df.empty:
                logger.warning(f"No data available for symbol {symbol}")
                return None
        else:
            df = evaluation_results.copy()
        
        # Filter by date range if specified
        if start_date is not None or end_date is not None:
            date_mask = True
            
            if start_date is not None:
                date_mask = date_mask & (df['date'] >= pd.Timestamp(start_date))
                
            if end_date is not None:
                date_mask = date_mask & (df['date'] <= pd.Timestamp(end_date))
                
            df = df[date_mask].copy()
        
        # Get actual volatility column based on estimator
        actual_col = f'actual_vol_{estimator.lower()}'
        
        if actual_col not in df.columns:
            logger.warning(f"Actual volatility column '{actual_col}' not found")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if symbol is not None:
            # Time series plot for a single symbol
            df = df.sort_values('date')
            ax.plot(df['date'], df[actual_col], label=f'Actual ({estimator})', linewidth=2)
            ax.plot(df['date'], df['pred_vol'], label='GARCH Prediction', linewidth=2, linestyle='--')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Volatility')
            ax.set_title(f'GARCH({self.p},{self.q}) Volatility Predictions for {symbol}')
            
            # Format date axis
            fig.autofmt_xdate()
            
        else:
            # Scatter plot for all symbols
            ax.scatter(df[actual_col], df['pred_vol'], alpha=0.5)
            
            # Add diagonal line (perfect predictions)
            min_val = min(df[actual_col].min(), df['pred_vol'].min())
            max_val = max(df[actual_col].max(), df['pred_vol'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--')
            
            ax.set_xlabel(f'Actual Volatility ({estimator})')
            ax.set_ylabel('Predicted Volatility')
            ax.set_title(f'GARCH({self.p},{self.q}) Volatility Predictions vs Actual')
            
            # Calculate R²
            r2 = self._calculate_r_squared(df['pred_vol'], df[actual_col])
            mse = df[f'squared_error_{estimator.lower()}'].mean()
            mae = df[f'abs_error_{estimator.lower()}'].mean()
            
            # Add metrics to plot
            ax.text(0.05, 0.95, f'R² = {r2:.4f}\nMSE = {mse:.6f}\nMAE = {mae:.6f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.grid(True)
        ax.legend()
        
        # Save figure if requested
        if save_path is not None:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        
        return fig