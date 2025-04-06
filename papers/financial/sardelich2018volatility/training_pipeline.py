import os
import logging
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import gensim.downloader as gensim_api

# Import our modules
from data_collection import (
    download_stock_prices,
    collect_reuters_headlines,
    compile_company_surface_forms,
    download_company_data_from_finviz
)
from data_processing import (
    calculate_volatility_estimators,
    calculate_price_features,
    preprocess_headlines,
    categorize_news_by_market_hours,
    align_news_with_prices,
    prepare_sliding_window_samples,
    create_embedding_matrix,
    split_data_by_time,
    build_vocabulary,
    analyze_data_statistics
)
from pytorch_dataset import VolatilityDataset, create_data_loaders
from garch_baseline import GARCHForecaster

# Import model implementations (you'll create this file next)
from models import (
    BiLSTMMaxPoolSentenceEncoder,
    BiLSTMAttentionSentenceEncoder,
    NewsRelevanceAttention,
    NewsTemporalContext,
    PriceEncoder,
    StockEmbedding,
    MultimodalVolatilityModel
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("volatility_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_config(config, filepath):
    """Save configuration to a JSON file"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Saved configuration to {filepath}")

def load_config(filepath):
    """Load configuration from a JSON file"""
    with open(filepath, 'r') as f:
        config = json.load(f)
    logger.info(f"Loaded configuration from {filepath}")
    return config

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                num_epochs, patience, device, checkpoint_dir, experiment_name):
    """
    Train the volatility prediction model with early stopping and checkpointing.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Maximum number of epochs
        patience: Patience for early stopping
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        experiment_name: Name of the experiment
        
    Returns:
        Trained model and training history
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    val_metrics = []
    
    # Start training
    logger.info(f"Starting training for {num_epochs} epochs")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            # Move data to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                headlines=batch['headlines'],
                headline_lengths=batch['headline_lengths'],
                headline_mask=batch['headline_mask'],
                news_mask=batch['headline_mask'],
                daily_news_mask=batch['daily_news_mask'],
                price_sequence=batch['price_features'],
                stock_indices=batch['stock_idx']
            )
            
            # Compute loss
            loss = criterion(outputs, batch['target_volatility'])
            
            # Backward pass and optimize
            loss.backward()
            
            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Accumulate loss
            train_loss += loss.item() * len(batch['stock_idx'])
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_outputs = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                # Move data to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    headlines=batch['headlines'],
                    headline_lengths=batch['headline_lengths'],
                    headline_mask=batch['headline_mask'],
                    news_mask=batch['headline_mask'],
                    daily_news_mask=batch['daily_news_mask'],
                    price_sequence=batch['price_features'],
                    stock_indices=batch['stock_idx']
                )
                
                # Compute loss
                loss = criterion(outputs, batch['target_volatility'])
                
                # Accumulate loss
                val_loss += loss.item() * len(batch['stock_idx'])
                
                # Collect predictions and targets for metrics
                all_outputs.append(outputs.cpu().numpy())
                all_targets.append(batch['target_volatility'].cpu().numpy())
        
        # Calculate average validation loss
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        # Calculate validation metrics
        all_outputs = np.concatenate(all_outputs, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        val_mse = np.mean((all_outputs - all_targets) ** 2)
        val_mae = np.mean(np.abs(all_outputs - all_targets))
        
        # Calculate R² (coefficient of determination)
        mean_target = np.mean(all_targets)
        ss_total = np.sum((all_targets - mean_target) ** 2)
        ss_residual = np.sum((all_targets - all_outputs) ** 2)
        val_r2 = 1 - (ss_residual / ss_total)
        
        val_metrics.append({
            'MSE': val_mse,
            'MAE': val_mae,
            'R²': val_r2
        })
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Calculate elapsed time
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        
        # Print progress
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {train_loss:.6f}, "
                   f"Val Loss: {val_loss:.6f}, "
                   f"Val MSE: {val_mse:.6f}, "
                   f"Val MAE: {val_mae:.6f}, "
                   f"Val R²: {val_r2:.4f}, "
                   f"Epoch Time: {epoch_time:.2f}s, "
                   f"Total Time: {total_time:.2f}s")
        
        # Check for best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            
            # Save best model
            checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics[-1],
                'train_loss': train_loss
            }, checkpoint_path)
            logger.info(f"Saved best model to {checkpoint_path}")
        else:
            early_stop_counter += 1
            
            if early_stop_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics[-1],
                'train_loss': train_loss
            }, checkpoint_path)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_metrics': val_metrics
    }
    
    history_path = os.path.join(checkpoint_dir, f"{experiment_name}_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    # Load best model
    best_checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_best.pth")
    if os.path.exists(best_checkpoint_path):
        checkpoint = torch.load(best_checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    return model, history

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device: Device to evaluate on
        
    Returns:
        Dictionary of evaluation metrics and predictions
    """
    model.eval()
    all_outputs = []
    all_targets = []
    all_symbols = []
    all_dates = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            # Move data to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                headlines=batch['headlines'],
                headline_lengths=batch['headline_lengths'],
                headline_mask=batch['headline_mask'],
                news_mask=batch['headline_mask'],
                daily_news_mask=batch['daily_news_mask'],
                price_sequence=batch['price_features'],
                stock_indices=batch['stock_idx']
            )
            
            # Collect predictions and targets
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(batch['target_volatility'].cpu().numpy())
            all_symbols.extend(batch['symbol'])
            all_dates.extend(batch['target_date'])
    
    # Concatenate all predictions and targets
    predictions = np.concatenate(all_outputs, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    # Calculate metrics
    mse = np.mean((predictions - targets) ** 2)
    mae = np.mean(np.abs(predictions - targets))
    
    # Calculate R² (coefficient of determination)
    mean_target = np.mean(targets)
    ss_total = np.sum((targets - mean_target) ** 2)
    ss_residual = np.sum((targets - predictions) ** 2)
    r_squared = 1 - (ss_residual / ss_total)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'symbol': all_symbols,
        'date': all_dates,
        'predicted': predictions.flatten(),
        'actual': targets.flatten(),
        'squared_error': (predictions.flatten() - targets.flatten()) ** 2,
        'abs_error': np.abs(predictions.flatten() - targets.flatten())
    })
    
    logger.info(f"Test Evaluation:")
    logger.info(f"  MSE: {mse:.6f}")
    logger.info(f"  MAE: {mae:.6f}")
    logger.info(f"  R²: {r_squared:.4f}")
    
    return {
        'MSE': mse,
        'MAE': mae,
        'R²': r_squared,
        'results_df': results_df
    }

def calculate_sector_metrics(results_df, symbol_to_sector):
    """
    Calculate evaluation metrics by sector.
    
    Args:
        results_df: DataFrame with evaluation results
        symbol_to_sector: Dictionary mapping symbols to sectors
        
    Returns:
        DataFrame with sector-level metrics
    """
    # Add sector column
    results_df['sector'] = results_df['symbol'].map(
        lambda x: symbol_to_sector.get(x, 'Unknown')
    )
    
    # Group by sector and calculate metrics
    sector_metrics = []
    
    for sector, group in results_df.groupby('sector'):
        # Calculate metrics
        mse = group['squared_error'].mean()
        mae = group['abs_error'].mean()
        
        # Calculate R²
        mean_target = group['actual'].mean()
        ss_total = ((group['actual'] - mean_target) ** 2).sum()
        ss_residual = ((group['actual'] - group['predicted']) ** 2).sum()
        r_squared = 1 - (ss_residual / ss_total)
        
        sector_metrics.append({
            'sector': sector,
            'num_predictions': len(group),
            'num_symbols': group['symbol'].nunique(),
            'MSE': mse,
            'MAE': mae,
            'R²': r_squared
        })
    
    return pd.DataFrame(sector_metrics)

def plot_training_history(history, save_path=None):
    """
    Plot training history.
    
    Args:
        history: Dictionary with training history
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training and validation loss
    axes[0, 0].plot(history['train_losses'], label='Train Loss')
    axes[0, 0].plot(history['val_losses'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot validation MSE
    axes[0, 1].plot([m['MSE'] for m in history['val_metrics']], label='MSE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].set_title('Validation MSE')
    axes[0, 1].grid(True)
    
    # Plot validation MAE
    axes[1, 0].plot([m['MAE'] for m in history['val_metrics']], label='MAE')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('MAE')
    axes[1, 0].set_title('Validation MAE')
    axes[1, 0].grid(True)
    
    # Plot validation R²
    axes[1, 1].plot([m['R²'] for m in history['val_metrics']], label='R²')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('R²')
    axes[1, 1].set_title('Validation R²')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training history plot to {save_path}")
    
    return fig

def plot_predictions(results_df, title="Model Predictions vs Actual", save_path=None):
    """
    Plot predicted vs actual volatility.
    
    Args:
        results_df: DataFrame with prediction results
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    scatter = ax.scatter(results_df['actual'], results_df['predicted'], alpha=0.5)
    
    # Add diagonal line (perfect predictions)
    min_val = min(results_df['actual'].min(), results_df['predicted'].min())
    max_val = max(results_df['actual'].max(), results_df['predicted'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        results_df['actual'], results_df['predicted']
    )
    ax.plot(
        [min_val, max_val], 
        [slope * min_val + intercept, slope * max_val + intercept], 
        'g-', label=f'Regression Line (slope={slope:.3f})'
    )
    
    # Calculate metrics
    mse = results_df['squared_error'].mean()
    mae = results_df['abs_error'].mean()
    r2 = r_value ** 2  # R² from regression
    
    # Add metrics to plot
    ax.text(0.05, 0.95, f'R² = {r2:.4f}\nMSE = {mse:.6f}\nMAE = {mae:.6f}',
           transform=ax.transAxes, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Actual Volatility')
    ax.set_ylabel('Predicted Volatility')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved predictions plot to {save_path}")
    
    return fig

def plot_sector_performance(sector_metrics, metric='R²', comparison_with=None, title=None, save_path=None):
    """
    Plot sector performance.
    
    Args:
        sector_metrics: DataFrame with sector metrics
        metric: Metric to plot ('MSE', 'MAE', or 'R²')
        comparison_with: Optional DataFrame with comparison metrics
        title: Plot title
        save_path: Path to save the plot
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sort by sector
    sector_metrics = sector_metrics.sort_values('sector')
    
    # Plot bars
    x = np.arange(len(sector_metrics))
    width = 0.35
    
    ax.bar(x - width/2 if comparison_with is not None else x, 
           sector_metrics[metric], width, label='Our Model')
    
    if comparison_with is not None:
        # Sort comparison data in the same order
        comparison_with = comparison_with.set_index('sector').loc[sector_metrics['sector']].reset_index()
        ax.bar(x + width/2, comparison_with[metric], width, label='GARCH(1,1)')
    
    # Add labels and title
    ax.set_xlabel('Sector')
    ax.set_ylabel(metric)
    ax.set_title(title or f'Sector Performance - {metric}')
    ax.set_xticks(x)
    ax.set_xticklabels(sector_metrics['sector'], rotation=45, ha='right')
    ax.legend()
    
    # Add grid
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}' if metric == 'R²' else f'{height:.6f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    
    # Add labels to the bars
    add_labels(ax.patches[:len(sector_metrics)])
    if comparison_with is not None:
        add_labels(ax.patches[len(sector_metrics):])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved sector performance plot to {save_path}")
    
    return fig

def main(config):
    """
    Main function to run the training and evaluation pipeline.
    
    Args:
        config: Configuration dictionary
    """
    # Set random seeds for reproducibility
    set_random_seeds(config['random_seed'])
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{config['experiment_name']}_{timestamp}"
    experiment_dir = os.path.join(config['output_dir'], experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save configuration
    save_config(config, os.path.join(experiment_dir, 'config.json'))
    
    # Define sectors and stocks
    sectors_stocks = config['sectors_stocks']
    
    # Get all unique stock symbols
    all_stocks = []
    for stocks in sectors_stocks.values():
        all_stocks.extend(stocks)
    all_stocks = list(set(all_stocks))  # Remove duplicates
    
    # Create symbol to sector mapping for later analysis
    symbol_to_sector = {}
    for sector, stocks in sectors_stocks.items():
        for symbol in stocks:
            symbol_to_sector[symbol] = sector
    
    # Step 1: Data Collection
    logger.info("Step 1: Data Collection")
    
    # Download price data
    price_data = download_stock_prices(
        all_stocks,
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        cache_dir=os.path.join(config['data']['cache_dir'], 'prices')
    )
    
    # Get company surface forms for news matching
    surface_forms = compile_company_surface_forms(all_stocks)
    
    # Collect news headlines
    headlines_data = collect_reuters_headlines(
        all_stocks,
        surface_forms,
        start_date=config['data']['start_date'],
        end_date=config['data']['end_date'],
        cache_dir=os.path.join(config['data']['cache_dir'], 'headlines')
    )
    
    # Optional: Get additional company information
    company_data = download_company_data_from_finviz(
        all_stocks,
        cache_dir=os.path.join(config['data']['cache_dir'], 'company_info')
    )
    
    # Step 2: Data Processing
    logger.info("Step 2: Data Processing")
    
    # Process price data
    processed_price_data = {}
    for symbol, df in price_data.items():
        if df is not None and not df.empty:
            # Calculate volatility estimators
            df_with_vol = calculate_volatility_estimators(df)
            # Calculate price features
            price_features = calculate_price_features(
                df_with_vol,
                include_additional_features=config['features']['include_additional_price_features']
            )
            processed_price_data[symbol] = price_features
    
    # Load word embeddings
    logger.info(f"Loading {config['word_embeddings']['name']} word embeddings...")
    word_embeddings = gensim_api.load(config['word_embeddings']['name'])
    
    # Process headlines
    processed_headlines = {}
    for symbol, headlines in headlines_data.items():
        if headlines is not None and not headlines.empty:
            # Categorize by market hours
            categorized_headlines = categorize_news_by_market_hours(headlines)
            # Preprocess headlines
            processed = preprocess_headlines(
                categorized_headlines,
                word_embeddings,
                min_words=config['features']['min_words_per_headline']
            )
            processed_headlines[symbol] = processed
    
    # Align news with prices
    aligned_news = align_news_with_prices(processed_headlines, processed_price_data)
    
    # Prepare sliding window samples
    samples = prepare_sliding_window_samples(
        aligned_news,
        processed_price_data,
        window_size=config['features']['window_size']
    )
    
    # Analyze dataset statistics
    data_stats = analyze_data_statistics(samples, processed_price_data, processed_headlines)
    
    # Save dataset statistics
    with open(os.path.join(experiment_dir, 'data_statistics.json'), 'w') as f:
        # Convert any non-serializable objects to strings
        serializable_stats = {}
        for key, value in data_stats.items():
            if isinstance(value, dict):
                serializable_stats[key] = {str(k): str(v) for k, v in value.items()}
            else:
                serializable_stats[key] = str(value)
        json.dump(serializable_stats, f, indent=4)
    
    # Build vocabulary
    word_to_idx = build_vocabulary(
        processed_headlines,
        min_freq=config['word_embeddings']['min_word_freq']
    )
    
    # Create embedding matrix
    embedding_matrix = create_embedding_matrix(word_to_idx, word_embeddings)
    
    # Split data
    train_samples, val_samples, test_samples = split_data_by_time(
        samples,
        train_end=config['data']['train_end'],
        val_end=config['data']['val_end']
    )
    
    # Step 3: Create Datasets and DataLoaders
    logger.info("Step 3: Creating datasets and dataloaders")
    
    # Create stock to index mapping
    stock_to_idx = {symbol: idx for idx, symbol in enumerate(all_stocks)}
    
    # Create datasets
    train_dataset = VolatilityDataset(
        train_samples,
        word_to_idx,
        stock_to_idx=stock_to_idx,
        max_words=config['model']['max_words'],
        max_news=config['model']['max_news'],
        window_size=config['features']['window_size'],
        cache_processed_data=True,
        cache_dir=os.path.join(config['data']['cache_dir'], 'processed')
    )
    
    val_dataset = VolatilityDataset(
        val_samples,
        word_to_idx,
        stock_to_idx=stock_to_idx,
        max_words=config['model']['max_words'],
        max_news=config['model']['max_news'],
        window_size=config['features']['window_size'],
        cache_processed_data=True,
        cache_dir=os.path.join(config['data']['cache_dir'], 'processed')
    )
    
    test_dataset = VolatilityDataset(
        test_samples,
        word_to_idx,
        stock_to_idx=stock_to_idx,
        max_words=config['model']['max_words'],
        max_news=config['model']['max_news'],
        window_size=config['features']['window_size'],
        cache_processed_data=True,
        cache_dir=os.path.join(config['data']['cache_dir'], 'processed')
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers']
    )
    
    # Log dataset sizes
    logger.info(f"Dataset sizes:")
    logger.info(f"  Train: {len(train_dataset)} samples")
    logger.info(f"  Validation: {len(val_dataset)} samples")
    logger.info(f"  Test: {len(test_dataset)} samples")
    
    # Step 4: Initialize Model
    logger.info("Step 4: Initializing model")
    
    # Determine input features dimension
    if config['features']['include_additional_price_features']:
        # This should match the features created in calculate_price_features()
        price_feature_dim = 10
    else:
        price_feature_dim = 4  # Basic price features from the paper
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not config['training']['force_cpu'] else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = MultimodalVolatilityModel(
        vocab_size=len(word_to_idx),
        embed_dim=config['word_embeddings']['dimension'],
        sentence_hidden_dim=config['model']['sentence_hidden_dim'],
        news_attention_dim=config['model']['news_attention_dim'],
        temporal_hidden_dim=config['model']['temporal_hidden_dim'],
        temporal_attention_dim=config['model']['temporal_attention_dim'],
        price_hidden_dim=config['model']['price_hidden_dim'],
        stock_embedding_dim=config['model']['stock_embedding_dim'],
        joint_dim=config['model']['joint_dim'],
        num_stocks=len(all_stocks),
        embedding_weights=embedding_matrix,
        price_feature_dim=price_feature_dim,
        use_bilstm_attention=config['model']['use_bilstm_attention'],
        use_news_relevance_attention=config['model']['use_news_relevance_attention']
    ).to(device)
    
    # Log model architecture
    logger.info(f"Model architecture:")
    logger.info(f"  {model}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters:")
    logger.info(f"  Total: {total_params:,}")
    logger.info(f"  Trainable: {trainable_params:,}")
    
    # Step 5: Train Model
    logger.info("Step 5: Training model")
    
    # Initialize optimizer
    if config['training']['optimizer'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    elif config['training']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']}")
    
    # Initialize loss function
    criterion = nn.MSELoss()
    
    # Initialize learning rate scheduler
    if config['training']['lr_scheduler'] == 'reduce_on_plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
    elif config['training']['lr_scheduler'] == 'cosine_annealing':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs'],
            eta_min=config['training']['min_lr']
        )
    elif config['training']['lr_scheduler'] == 'one_cycle':
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config['training']['learning_rate'],
            steps_per_epoch=len(train_loader),
            epochs=config['training']['num_epochs']
        )
    else:
        scheduler = None
    
    # Train the model
    trained_model, history = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        num_epochs=config['training']['num_epochs'],
        patience=config['training']['patience'],
        device=device,
        checkpoint_dir=os.path.join(experiment_dir, 'checkpoints'),
        experiment_name=experiment_name
    )
    
    # Plot training history
    plot_training_history(
        history,
        save_path=os.path.join(experiment_dir, 'training_history.png')
    )
    
    # Step 6: Evaluate on Test Set
    logger.info("Step 6: Evaluating on test set")
    
    # Evaluate model
    test_results = evaluate_model(trained_model, test_loader, device)
    
    # Save test results
    test_results['results_df'].to_csv(os.path.join(experiment_dir, 'test_results.csv'), index=False)
    
    # Calculate sector metrics
    sector_metrics = calculate_sector_metrics(test_results['results_df'], symbol_to_sector)
    sector_metrics.to_csv(os.path.join(experiment_dir, 'sector_metrics.csv'), index=False)
    
    # Plot predictions
    plot_predictions(
        test_results['results_df'],
        title=f"Model Predictions vs Actual (Test Set, R²={test_results['R²']:.4f})",
        save_path=os.path.join(experiment_dir, 'test_predictions.png')
    )
    
    # Step 7: Compare with GARCH Baseline
    logger.info("Step 7: Comparing with GARCH baseline")
    
    # Initialize and fit GARCH models
    garch_forecaster = GARCHForecaster(
        p=1,
        q=1,
        cache_dir=os.path.join(config['data']['cache_dir'], 'garch'),
        use_cache=True
    )
    
    # Get symbols from test set
    test_symbols = test_results['results_df']['symbol'].unique()
    
    # Fit GARCH models
    garch_forecaster.fit(
        price_data,
        symbols=test_symbols,
        max_workers=config['training']['num_workers']
    )
    
    # Evaluate GARCH models
    garch_results, garch_metrics = garch_forecaster.evaluate(
        price_data,
        start_date=config['data']['val_end'],  # Start from the validation end date (beginning of test set)
        end_date=config['data']['end_date']
    )
    
    # Save GARCH results
    garch_results.to_csv(os.path.join(experiment_dir, 'garch_results.csv'), index=False)
    
    # Calculate GARCH sector metrics
    garch_sector_metrics = garch_forecaster.calculate_sector_metrics(garch_results, symbol_to_sector)
    garch_sector_metrics.to_csv(os.path.join(experiment_dir, 'garch_sector_metrics.csv'), index=False)
    
    # Plot GARCH predictions
    garch_forecaster.plot_predictions(
        garch_results,
        estimator='GK',
        save_path=os.path.join(experiment_dir, 'garch_predictions.png')
    )
    
    # Step 8: Compare performance across sectors
    logger.info("Step 8: Comparing sector performance")
    
    # Plot sector performance comparison
    for metric in ['MSE', 'MAE', 'R²']:
        plot_sector_performance(
            sector_metrics,
            metric=metric,
            comparison_with=garch_sector_metrics,
            title=f"Sector Performance - {metric} (Our Model vs GARCH)",
            save_path=os.path.join(experiment_dir, f'sector_comparison_{metric}.png')
        )
    
    # Step 9: Save Summary Report
    logger.info("Step 9: Saving summary report")
    
    # Prepare summary data
    summary = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'model_metrics': {
            'MSE': test_results['MSE'],
            'MAE': test_results['MAE'],
            'R²': test_results['R²']
        },
        'garch_metrics': {
            'MSE': garch_metrics['GK']['MSE'],
            'MAE': garch_metrics['GK']['MAE'],
            'R²': garch_metrics['GK']['R²']
        },
        'improvement': {
            'MSE': (garch_metrics['GK']['MSE'] - test_results['MSE']) / garch_metrics['GK']['MSE'] * 100,
            'MAE': (garch_metrics['GK']['MAE'] - test_results['MAE']) / garch_metrics['GK']['MAE'] * 100,
            'R²': (test_results['R²'] - garch_metrics['GK']['R²']) / garch_metrics['GK']['R²'] * 100
        },
        'sector_metrics': sector_metrics.to_dict(orient='records'),
        'garch_sector_metrics': garch_sector_metrics.to_dict(orient='records')
    }
    
    # Save summary
    with open(os.path.join(experiment_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Log summary
    logger.info("Performance Summary:")
    logger.info(f"  Our Model: MSE={test_results['MSE']:.6f}, MAE={test_results['MAE']:.6f}, R²={test_results['R²']:.4f}")
    logger.info(f"  GARCH(1,1): MSE={garch_metrics['GK']['MSE']:.6f}, MAE={garch_metrics['GK']['MAE']:.6f}, R²={garch_metrics['GK']['R²']:.4f}")
    logger.info(f"  Improvement: MSE={summary['improvement']['MSE']:.2f}%, MAE={summary['improvement']['MAE']:.2f}%, R²={summary['improvement']['R²']:.2f}%")
    
    logger.info(f"Experiment complete. Results saved to {experiment_dir}")
    return summary

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Volatility Prediction Training Pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run main function
    main(config)