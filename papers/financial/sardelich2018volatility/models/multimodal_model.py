import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import logging

# Set up logging
logger = logging.getLogger(__name__)

class BiLSTMMaxPoolSentenceEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, embedding_weights=None, dropout=0.2):
        """
        BiLSTM with Max Pooling sentence encoder.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden states
            embedding_weights: Optional pretrained embedding weights
            dropout: Dropout probability
        """
        super(BiLSTMMaxPoolSentenceEncoder, self).__init__()
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Initialize with pretrained embeddings if provided
        if embedding_weights is not None:
            embedding_weights = torch.FloatTensor(embedding_weights)
            self.embedding.weight.data.copy_(embedding_weights)
            # Don't fine-tune the word embeddings (as in the paper)
            self.embedding.weight.requires_grad = False
            
        # BiLSTM layer
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            bidirectional=True, 
            batch_first=True,
            dropout=dropout if dropout > 0 and dropout < 1 else 0
        )
        
        # Output dimension
        self.output_dim = hidden_dim * 2  # Bidirectional
    
    def forward(self, x, lengths):
        """
        Args:
            x: Tensor of word indices [batch_size, max_seq_len]
            lengths: List/Tensor of actual sequence lengths
            
        Returns:
            Sentence embedding [batch_size, output_dim]
        """
        # Handle empty input (all lengths are zero)
        if torch.all(lengths == 0):
            return torch.zeros(x.size(0), self.output_dim, device=x.device)
            
        # Handle batch with all padded sequences
        if torch.all(lengths <= 0):
            logger.warning("All sequences have length <= 0, returning zero embeddings")
            return torch.zeros(x.size(0), self.output_dim, device=x.device)
        
        # Filter out empty sequences
        valid_indices = torch.where(lengths > 0)[0]
        if len(valid_indices) == 0:
            return torch.zeros(x.size(0), self.output_dim, device=x.device)
            
        valid_x = x[valid_indices]
        valid_lengths = lengths[valid_indices]
        
        # Get word embeddings
        embedded = self.embedding(valid_x)  # [valid_batch_size, max_seq_len, embed_dim]
        
        # Apply dropout to embeddings
        embedded = self.dropout(embedded)
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, valid_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Process with BiLSTM
        packed_output, _ = self.lstm(packed)  # [valid_batch_size, max_seq_len, 2*hidden_dim]
        
        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Attention mechanism (Equations 29-31 in the paper)
        # Transform hidden states
        attention_hidden = torch.tanh(self.attention_layer(output))  # [valid_batch_size, max_seq_len, attention_dim]
        
        # Calculate attention scores
        attention_scores = torch.matmul(attention_hidden, self.attention_vector)  # [valid_batch_size, max_seq_len, 1]
        
        # Mask padded positions
        mask = torch.arange(output.size(1), device=x.device)[None, :] < valid_lengths[:, None]
        attention_scores = attention_scores.masked_fill(~mask.unsqueeze(2), float('-inf'))
        
        # Softmax for attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [valid_batch_size, max_seq_len, 1]
        
        # Apply attention weights
        valid_sentence_embedding = torch.sum(output * attention_weights, dim=1)  # [valid_batch_size, 2*hidden_dim]
        
        # Create full batch result (initialize with zeros for invalid sequences)
        sentence_embedding = torch.zeros(x.size(0), self.output_dim, device=x.device)
        sentence_embedding[valid_indices] = valid_sentence_embedding
        
        return sentence_embedding

class NewsRelevanceAttention(nn.Module):
    def __init__(self, sentence_dim, attention_dim, dropout=0.2):
        """
        News Relevance Attention (NRA) to highlight the most relevant news on a given day.
        
        Args:
            sentence_dim: Dimension of sentence embeddings
            attention_dim: Dimension of attention hidden layer
            dropout: Dropout probability
        """
        super(NewsRelevanceAttention, self).__init__()
        
        # Attention layers
        self.attention_layer = nn.Linear(sentence_dim, attention_dim)
        self.attention_vector = nn.Parameter(torch.Tensor(attention_dim, 1))
        nn.init.xavier_uniform_(self.attention_vector)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, news_embeddings, mask=None):
        """
        Args:
            news_embeddings: Tensor of news sentence embeddings [batch_size, max_news, sentence_dim]
            mask: Boolean mask for valid news [batch_size, max_news]
            
        Returns:
            Daily news embedding [batch_size, sentence_dim]
        """
        # Apply dropout
        news_embeddings = self.dropout(news_embeddings)
        
        # Transform news embeddings
        attention_hidden = torch.tanh(self.attention_layer(news_embeddings))  # [batch_size, max_news, attention_dim]
        
        # Calculate attention scores
        attention_scores = torch.matmul(attention_hidden, self.attention_vector)  # [batch_size, max_news, 1]
        
        # Mask invalid news if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask.unsqueeze(2), float('-inf'))
        
        # Softmax for attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, max_news, 1]
        
        # Apply attention weights
        daily_news_embedding = torch.sum(news_embeddings * attention_weights, dim=1)  # [batch_size, sentence_dim]
        
        return daily_news_embedding

class ZerosImputation(nn.Module):
    def __init__(self, embedding_dim):
        """
        Zeros & Imputation module for handling missing news days.
        
        Args:
            embedding_dim: Dimension of news embeddings
        """
        super(ZerosImputation, self).__init__()
        self.embedding_dim = embedding_dim
    
    def forward(self, daily_news_embeddings, daily_news_mask):
        """
        Implement the Zeros & Imputation method from the paper.
        
        Args:
            daily_news_embeddings: Tensor of daily news embeddings [batch_size, max_days, embedding_dim]
            daily_news_mask: Boolean mask for days with news [batch_size, max_days]
            
        Returns:
            Processed embeddings with missingness indicator [batch_size, max_days, embedding_dim*2]
        """
        batch_size, max_days, embedding_dim = daily_news_embeddings.shape
        
        # Create missingness indicator (1 for missing, 0 for present)
        missingness = (~daily_news_mask).float().unsqueeze(-1)  # [batch_size, max_days, 1]
        
        # Zero out embeddings for days without news
        processed_embeddings = daily_news_embeddings * daily_news_mask.unsqueeze(-1).float()
        
        # Expand indicator feature to match embedding dimension
        expanded_indicator = missingness.expand(-1, -1, embedding_dim)
        
        # Concatenate the processed embeddings and indicator
        result = torch.cat([processed_embeddings, expanded_indicator], dim=-1)
        
        return result

class NewsTemporalContext(nn.Module):
    def __init__(self, input_dim, hidden_dim, attention_dim, dropout=0.2):
        """
        News Temporal Context layer using BiLSTM with attention.
        
        Args:
            input_dim: Dimension of input daily news embeddings
            hidden_dim: Dimension of LSTM hidden states
            attention_dim: Dimension of attention hidden layer
            dropout: Dropout probability
        """
        super(NewsTemporalContext, self).__init__()
        
        # BiLSTM layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            bidirectional=True, 
            batch_first=True,
            dropout=dropout if dropout > 0 and dropout < 1 else 0
        )
        
        # Attention layers
        self.attention_layer = nn.Linear(2 * hidden_dim, attention_dim)
        self.attention_vector = nn.Parameter(torch.Tensor(attention_dim, 1))
        nn.init.xavier_uniform_(self.attention_vector)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension
        self.output_dim = hidden_dim * 2  # Bidirectional
    
    def forward(self, daily_news_sequence, mask=None):
        """
        Args:
            daily_news_sequence: Tensor of daily news embeddings [batch_size, seq_len, input_dim]
            mask: Boolean mask for valid days [batch_size, seq_len]
            
        Returns:
            Market news embedding [batch_size, output_dim]
        """
        # Apply dropout
        daily_news_sequence = self.dropout(daily_news_sequence)
        
        # Process with BiLSTM
        output, _ = self.lstm(daily_news_sequence)  # [batch_size, seq_len, 2*hidden_dim]
        
        # Apply dropout to LSTM output
        output = self.dropout(output)
        
        # Attention mechanism
        attention_hidden = torch.tanh(self.attention_layer(output))  # [batch_size, seq_len, attention_dim]
        
        # Calculate attention scores
        attention_scores = torch.matmul(attention_hidden, self.attention_vector)  # [batch_size, seq_len, 1]
        
        # Mask invalid days if provided
        if mask is not None:
            attention_scores = attention_scores.masked_fill(~mask.unsqueeze(2), float('-inf'))
        
        # Softmax for attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len, 1]
        
        # Apply attention weights
        market_news_embedding = torch.sum(output * attention_weights, dim=1)  # [batch_size, 2*hidden_dim]
        
        return market_news_embedding

class PriceEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.2):
        """
        Price Encoder using stacked LSTMs.
        
        Args:
            input_dim: Dimension of price features
            hidden_dim: Dimension of LSTM hidden states
            num_layers: Number of stacked LSTM layers
            dropout: Dropout probability
        """
        super(PriceEncoder, self).__init__()
        
        # First LSTM layer
        self.lstm1 = nn.LSTM(
            input_dim, 
            hidden_dim, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Second LSTM layer
        self.lstm2 = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            batch_first=True
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Store parameters
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Output dimension
        self.output_dim = hidden_dim
    
    def forward(self, price_sequence):
        """
        Args:
            price_sequence: Tensor of price features [batch_size, seq_len, input_dim]
            
        Returns:
            Market price embedding [batch_size, output_dim]
        """
        # Apply dropout to input
        price_sequence = self.dropout(price_sequence)
        
        # Process with first LSTM
        output, _ = self.lstm1(price_sequence)  # [batch_size, seq_len, hidden_dim]
        
        # Apply dropout between LSTM layers
        output = self.dropout(output)
        
        # Process with second LSTM
        _, (h_n, _) = self.lstm2(output)  # h_n: [1, batch_size, hidden_dim]
        
        # Extract the final hidden state
        market_price_embedding = h_n.squeeze(0)  # [batch_size, hidden_dim]
        
        return market_price_embedding

class StockEmbedding(nn.Module):
    def __init__(self, num_stocks, embedding_dim, dropout=0.2):
        """
        Stock Embedding layer.
        
        Args:
            num_stocks: Number of unique stocks
            embedding_dim: Dimension of stock embeddings
            dropout: Dropout probability
        """
        super(StockEmbedding, self).__init__()
        
        # Stock embedding layer
        self.embedding = nn.Embedding(num_stocks, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension
        self.output_dim = embedding_dim
    
    def forward(self, stock_indices):
        """
        Args:
            stock_indices: Tensor of stock indices [batch_size]
            
        Returns:
            Stock embeddings [batch_size, output_dim]
        """
        # Get embeddings
        embeddings = self.embedding(stock_indices)  # [batch_size, embedding_dim]
        
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings

class MultimodalVolatilityModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, sentence_hidden_dim, news_attention_dim, 
                 temporal_hidden_dim, temporal_attention_dim, price_hidden_dim, 
                 stock_embedding_dim, joint_dim, num_stocks, embedding_weights=None,
                 price_feature_dim=4, use_bilstm_attention=True, use_news_relevance_attention=True,
                 dropout=0.2):
        """
        Complete Multimodal Volatility Prediction Model.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of word embeddings
            sentence_hidden_dim: Dimension of sentence LSTM hidden states
            news_attention_dim: Dimension of news attention hidden layer
            temporal_hidden_dim: Dimension of temporal LSTM hidden states
            temporal_attention_dim: Dimension of temporal attention hidden layer
            price_hidden_dim: Dimension of price LSTM hidden states
            stock_embedding_dim: Dimension of stock embeddings
            joint_dim: Dimension of joint representation
            num_stocks: Number of unique stocks
            embedding_weights: Optional pretrained embedding weights
            price_feature_dim: Dimension of price features
            use_bilstm_attention: Whether to use BiLSTM attention for sentence encoding
            use_news_relevance_attention: Whether to use news relevance attention
            dropout: Dropout probability
        """
        super(MultimodalVolatilityModel, self).__init__()
        
        # Store parameters
        self.use_bilstm_attention = use_bilstm_attention
        self.use_news_relevance_attention = use_news_relevance_attention
        
        # 1. Sentence Encoder
        if use_bilstm_attention:
            self.sentence_encoder = BiLSTMAttentionSentenceEncoder(
                vocab_size, embed_dim, sentence_hidden_dim, news_attention_dim, 
                embedding_weights, dropout
            )
        else:
            self.sentence_encoder = BiLSTMMaxPoolSentenceEncoder(
                vocab_size, embed_dim, sentence_hidden_dim, embedding_weights, dropout
            )
        
        # Output dimensions from sentence encoder
        sentence_dim = 2 * sentence_hidden_dim  # Bidirectional
        
        # 2. News Relevance Attention
        if use_news_relevance_attention:
            self.news_relevance_attention = NewsRelevanceAttention(
                sentence_dim, news_attention_dim, dropout
            )
        
        # 3. Zeros & Imputation
        self.zeros_imputation = ZerosImputation(sentence_dim)
        
        # 4. News Temporal Context
        # Input dimension is doubled due to concatenation of indicator features
        self.news_temporal_context = NewsTemporalContext(
            sentence_dim * 2, temporal_hidden_dim, temporal_attention_dim, dropout
        )
        
        # 5. Price Encoder
        self.price_encoder = PriceEncoder(price_feature_dim, price_hidden_dim, dropout=dropout)
        
        # 6. Stock Embedding
        self.stock_embedding = StockEmbedding(num_stocks, stock_embedding_dim, dropout)
        
        # 7. Joint Representation
        news_dim = self.news_temporal_context.output_dim
        price_dim = self.price_encoder.output_dim
        stock_dim = self.stock_embedding.output_dim
        
        joint_input_dim = news_dim + price_dim + stock_dim
        
        self.joint_representation = nn.Sequential(
            nn.Linear(joint_input_dim, joint_dim),
            nn.BatchNorm1d(joint_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 8. Volatility Prediction
        self.volatility_predictor = nn.Linear(joint_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for linear layers"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, headlines, headline_lengths, headline_mask, news_mask, 
                daily_news_mask, price_sequence, stock_indices):
        """
        Args:
            headlines: Tensor of headline word indices [batch_size, max_days, max_news, max_words]
            headline_lengths: Tensor of headline lengths [batch_size, max_days, max_news]
            headline_mask: Boolean mask for headlines [batch_size, max_days, max_news]
            news_mask: Boolean mask for news presence [batch_size, max_days, max_news]
            daily_news_mask: Boolean mask for days with news [batch_size, max_days]
            price_sequence: Tensor of price features [batch_size, max_days, price_feature_dim]
            stock_indices: Tensor of stock indices [batch_size]
            
        Returns:
            Predicted volatility [batch_size, 1]
        """
        batch_size, max_days, max_news, max_words = headlines.shape
        
        # 1. Encode each headline
        daily_news_embeddings = torch.zeros(
            batch_size, max_days, self.sentence_encoder.output_dim, 
            device=headlines.device
        )
        
        for day_idx in range(max_days):
            # Process headlines for this day if there are any
            if torch.any(daily_news_mask[:, day_idx]):
                # Get headlines for this day
                day_headlines = headlines[:, day_idx]  # [batch_size, max_news, max_words]
                day_lengths = headline_lengths[:, day_idx]  # [batch_size, max_news]
                day_mask = headline_mask[:, day_idx]  # [batch_size, max_news]
                
                # Iterate through headlines in this day
                day_embeddings = torch.zeros(
                    batch_size, max_news, self.sentence_encoder.output_dim,
                    device=headlines.device
                )
                
                for news_idx in range(max_news):
                    # Check if there are any valid headlines at this position
                    if torch.any(day_mask[:, news_idx]):
                        # Get headlines at this position
                        current_headlines = day_headlines[:, news_idx]  # [batch_size, max_words]
                        current_lengths = day_lengths[:, news_idx]  # [batch_size]
                        
                        # Encode headlines
                        headline_embeddings = self.sentence_encoder(current_headlines, current_lengths)
                        
                        # Store embeddings
                        day_embeddings[:, news_idx] = headline_embeddings
                
                # 2. Apply news relevance attention to get daily news embedding
                if self.use_news_relevance_attention:
                    day_embedding = self.news_relevance_attention(day_embeddings, day_mask)
                else:
                    # Simple averaging (as in the baseline)
                    # Use mask to avoid division by zero
                    mask_sum = day_mask.sum(dim=1, keepdim=True)
                    mask_sum = torch.clamp(mask_sum, min=1)  # Avoid division by zero
                    day_embedding = (day_embeddings * day_mask.unsqueeze(-1)).sum(dim=1) / mask_sum
                
                # Store daily news embedding
                daily_news_embeddings[:, day_idx] = day_embedding
        
        # 3. Apply Zeros & Imputation for missing news days
        processed_daily_news = self.zeros_imputation(daily_news_embeddings, daily_news_mask)
        
        # 4. Apply news temporal context
        market_news_embeddings = self.news_temporal_context(processed_daily_news)
        
        # 5. Process price sequence
        market_price_embeddings = self.price_encoder(price_sequence)
        
        # 6. Get stock embeddings
        stock_embeddings = self.stock_embedding(stock_indices)
        
        # 7. Joint representation
        joint_input = torch.cat([market_news_embeddings, market_price_embeddings, stock_embeddings], dim=1)
        joint_representation = self.joint_representation(joint_input)
        
        # 8. Predict volatility
        predicted_volatility = self.volatility_predictor(joint_representation)
        
        return predicted_volatility with all padded sequences
        if torch.all(lengths <= 0):
            logger.warning("All sequences have length <= 0, returning zero embeddings")
            return torch.zeros(x.size(0), self.output_dim, device=x.device)
        
        # Filter out empty sequences
        valid_indices = torch.where(lengths > 0)[0]
        if len(valid_indices) == 0:
            return torch.zeros(x.size(0), self.output_dim, device=x.device)
            
        valid_x = x[valid_indices]
        valid_lengths = lengths[valid_indices]
        
        # Get word embeddings
        embedded = self.embedding(valid_x)  # [valid_batch_size, max_seq_len, embed_dim]
        
        # Pack padded sequence
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, valid_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        
        # Process with BiLSTM
        packed_output, _ = self.lstm(packed)  # [valid_batch_size, max_seq_len, 2*hidden_dim]
        
        # Unpack
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        # Max pooling over sequence dimension
        valid_sentence_embedding = torch.max(output, dim=1)[0]  # [valid_batch_size, 2*hidden_dim]
        
        # Create full batch result (initialize with zeros for invalid sequences)
        sentence_embedding = torch.zeros(x.size(0), self.output_dim, device=x.device)
        sentence_embedding[valid_indices] = valid_sentence_embedding
        
        return sentence_embedding

class BiLSTMAttentionSentenceEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, attention_dim, embedding_weights=None, dropout=0.2):
        """
        BiLSTM with Attention sentence encoder.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of word embeddings
            hidden_dim: Dimension of LSTM hidden states
            attention_dim: Dimension of attention hidden layer
            embedding_weights: Optional pretrained embedding weights
            dropout: Dropout probability
        """
        super(BiLSTMAttentionSentenceEncoder, self).__init__()
        
        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Initialize with pretrained embeddings if provided
        if embedding_weights is not None:
            embedding_weights = torch.FloatTensor(embedding_weights)
            self.embedding.weight.data.copy_(embedding_weights)
            # Don't fine-tune the word embeddings (as in the paper)
            self.embedding.weight.requires_grad = False
            
        # BiLSTM layer
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_dim, 
            bidirectional=True, 
            batch_first=True,
            dropout=dropout if dropout > 0 and dropout < 1 else 0
        )
        
        # Attention layers
        self.attention_layer = nn.Linear(2 * hidden_dim, attention_dim)
        self.attention_vector = nn.Parameter(torch.Tensor(attention_dim, 1))
        nn.init.xavier_uniform_(self.attention_vector)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Output dimension
        self.output_dim = hidden_dim * 2  # Bidirectional
    
    def forward(self, x, lengths):
        """
        Args:
            x: Tensor of word indices [batch_size, max_seq_len]
            lengths: List/Tensor of actual sequence lengths
            
        Returns:
            Sentence embedding [batch_size, output_dim]
        """
        # Handle empty input (all lengths are zero)
        if torch.all(lengths == 0):
            return torch.zeros(x.size(0), self.output_dim, device=x.device)
            
        # Handle batch