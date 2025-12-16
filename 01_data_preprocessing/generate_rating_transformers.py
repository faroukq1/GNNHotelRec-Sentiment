import pandas as pd
import torch
import numpy as np
from transformers import pipeline
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import gc
import warnings
import os
warnings.filterwarnings('ignore')

# Set environment variables for stability
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'

# %%
# Load data
print("📂 Loading dataset...")
df = pd.read_csv('../data/tokenized/data_cleaned_full_with_tokens.csv')
print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")

# Prepare texts
if 'combined_text' in df.columns:
    texts = df['combined_text'].fillna('').astype(str).tolist()
else:
    def combine_text(row):
        title = row.get('review_title_clean', row.get('review_title', ''))
        text = row.get('review_text_clean', row.get('review_text', ''))
        if pd.isna(title): title = ''
        if pd.isna(text): text = ''
        return f"{title} {text}".strip()
    
    texts = df.apply(combine_text, axis=1).tolist()
    df['combined_text'] = texts

print(f"📝 Texts prepared: {len(texts):,}")

# %%
class CamelLabScorer:
    """CAMeL-Lab sentiment scorer (0-10 scale)"""
    
    def __init__(self, num_threads=12):
        self.num_threads = num_threads
        print(f"⚡ CAMeL-Lab Scorer initialized with {num_threads} threads")
        
        # Initialize the model
        print("🔄 Loading CAMeL-Lab model...")
        self.model = pipeline(
            "sentiment-analysis",
            model="CAMeL-Lab/bert-base-arabic-camelbert-mix-sentiment",
            device=-1,  # CPU (more stable for multi-threading)
            truncation=True,
            max_length=256  # Reasonable length for accuracy
        )
        print("✅ Model loaded successfully")
    
    def convert_to_10_point_scale(self, label, confidence_score):
        """
        Convert sentiment prediction to 0-10 scale
        
        CAMeL-Lab outputs:
        - LABEL_0: Negative (سلبي)
        - LABEL_1: Positive (إيجابي)
        
        We'll map:
        - Negative (LABEL_0): 0-4.9 based on confidence
        - Positive (LABEL_1): 5-10 based on confidence
        """
        
        # Map labels
        label_str = str(label).upper()
        
        if 'LABEL_1' in label_str or 'POSITIVE' in label_str.upper() or 'إيجابي' in label_str:
            # Positive sentiment: Map confidence to 5-10
            # confidence_score is between 0.5 and 1.0 for confident predictions
            # We'll scale it to 5-10 range
            base_score = 5.0
            score_range = 5.0  # From 5 to 10 = 5 points range
            
            # Enhanced mapping for positive sentiment
            if confidence_score > 0.95:
                return 10.0
            elif confidence_score > 0.9:
                return 9.0 + (confidence_score - 0.9) * 10  # 9.0-9.5
            elif confidence_score > 0.8:
                return 8.0 + (confidence_score - 0.8) * 10  # 8.0-9.0
            elif confidence_score > 0.7:
                return 7.0 + (confidence_score - 0.7) * 10  # 7.0-8.0
            elif confidence_score > 0.6:
                return 6.0 + (confidence_score - 0.6) * 10  # 6.0-7.0
            else:
                return 5.0 + (confidence_score - 0.5) * 10  # 5.0-6.0
                
        else:  # Negative sentiment (LABEL_0 or other)
            # Negative sentiment: Map confidence to 0-4.9
            # Higher confidence in negative = lower score
            base_score = 0.0
            score_range = 4.9  # From 0 to 4.9
            
            # Enhanced mapping for negative sentiment
            if confidence_score > 0.95:
                return 0.0
            elif confidence_score > 0.9:
                return 0.5 - (confidence_score - 0.9) * 5  # 0.0-0.5
            elif confidence_score > 0.8:
                return 1.0 - (confidence_score - 0.8) * 5  # 0.5-1.0
            elif confidence_score > 0.7:
                return 2.0 - (confidence_score - 0.7) * 10  # 1.0-2.0
            elif confidence_score > 0.6:
                return 3.0 - (confidence_score - 0.6) * 10  # 2.0-3.0
            else:
                return 4.0 - (confidence_score - 0.5) * 10  # 3.0-4.0
    
    def process_batch(self, batch_texts):
        """Process a batch of texts"""
        try:
            # Get predictions from model
            predictions = self.model(batch_texts)
            
            scores = []
            for pred in predictions:
                label = pred['label']
                confidence = pred['score']
                
                # Convert to 0-10 scale
                score = self.convert_to_10_point_scale(label, confidence)
                
                # Round to 1 decimal place for readability
                scores.append(round(score, 1))
            
            return scores
            
        except Exception as e:
            # If error, return neutral scores (5.0)
            # print(f"Batch error: {e}")
            return [5.0] * len(batch_texts)
    
    def score_texts_parallel(self, texts, batch_size=64):
        """
        Score all texts in parallel using multiple threads
        
        Args:
            texts: List of texts to score
            batch_size: Number of texts per batch
        
        Returns:
            List of scores (0-10 scale)
        """
        print(f"\n🎯 Starting parallel scoring...")
        print(f"📊 Total texts: {len(texts):,}")
        print(f"📦 Batch size: {batch_size}")
        print(f"🧵 Threads: {self.num_threads}")
        
        start_time = time.time()
        
        # Split texts into batches
        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batches.append((i // batch_size, batch))
        
        print(f"📋 Total batches: {len(batches):,}")
        
        # Initialize results array
        scores = [5.0] * len(texts)  # Default neutral score
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all batches
            future_to_batch = {
                executor.submit(self.process_batch, batch): batch_idx
                for batch_idx, batch in batches
            }
            
            # Collect results with progress bar
            completed = 0
            with tqdm(total=len(batches), desc="Scoring Progress") as pbar:
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    
                    try:
                        batch_scores = future.result()
                        
                        # Calculate start index for this batch
                        start_idx = batch_idx * batch_size
                        end_idx = start_idx + len(batch_scores)
                        
                        # Update scores array
                        scores[start_idx:end_idx] = batch_scores
                        
                        completed += 1
                        
                        # Update progress bar every 10 batches
                        if completed % 10 == 0:
                            pbar.update(10)
                        
                    except Exception as e:
                        print(f"\n⚠️ Error in batch {batch_idx}: {e}")
                        # Keep default score of 5.0 for this batch
                    
            # Update progress bar to 100%
            pbar.update(len(batches) - completed)
        
        # Calculate statistics
        elapsed_time = time.time() - start_time
        avg_score = np.mean(scores)
        score_std = np.std(scores)
        
        print(f"\n{'='*60}")
        print("📈 SCORING COMPLETE!")
        print("="*60)
        print(f"⏱️  Time: {elapsed_time:.1f} seconds ({elapsed_time/60:.1f} minutes)")
        print(f"⚡ Speed: {len(texts)/elapsed_time:.1f} texts/second")
        print(f"📊 Average score: {avg_score:.2f}/10")
        print(f"📐 Score std: {score_std:.2f}")
        print(f"🔢 Score range: {min(scores):.1f} - {max(scores):.1f}")
        print("="*60)
        
        return scores
    
    def analyze_sample(self, sample_texts=10):
        """Analyze sample texts to show scoring logic"""
        print("\n" + "="*60)
        print("🔍 SAMPLE ANALYSIS - Scoring Logic")
        print("="*60)
        
        # Get first few texts
        sample = texts[:sample_texts]
        
        for i, text in enumerate(sample):
            if text.strip():
                print(f"\n📝 Text {i+1}: {text[:100]}...")
                prediction = self.model(text)[0]
                label = prediction['label']
                confidence = prediction['score']
                score = self.convert_to_10_point_scale(label, confidence)
                
                print(f"   Label: {label}")
                print(f"   Confidence: {confidence:.3f}")
                print(f"   Score (0-10): {score:.1f}")
            else:
                print(f"\n📝 Text {i+1}: [EMPTY]")
                print(f"   Score (0-10): 5.0 (default for empty)")

# %%
# Initialize scorer
print("\n" + "="*80)
print("🎯 CAMeL-LAB SENTIMENT SCORER (0-10 Scale)")
print("="*80)

scorer = CamelLabScorer(num_threads=12)

# %%
# Show sample analysis
scorer.analyze_sample(sample_texts=5)

# %%
# Run scoring on all texts
print("\n" + "="*80)
print("🚀 STARTING FULL DATASET SCORING")
print("="*80)

scores = scorer.score_texts_parallel(texts, batch_size=32)

# Add scores to dataframe
df['camelbert_sentiment_score'] = scores

# %%
# Create sentiment category based on score
def score_to_category(score):
    """Convert 0-10 score to sentiment category"""
    if score >= 8.0:
        return "Very Positive"
    elif score >= 6.0:
        return "Positive"
    elif score >= 4.0:
        return "Neutral"
    elif score >= 2.0:
        return "Negative"
    else:
        return "Very Negative"

df['camelbert_sentiment_category'] = df['camelbert_sentiment_score'].apply(score_to_category)

# %%
# Display detailed statistics
print("\n" + "="*80)
print("📊 DETAILED STATISTICS")
print("="*80)

# Score distribution
print("\n📈 Score Distribution (0-10 scale):")
score_stats = df['camelbert_sentiment_score'].describe()
print(score_stats)

# Category distribution
print("\n🎯 Sentiment Category Distribution:")
category_counts = df['camelbert_sentiment_category'].value_counts()
for category, count in category_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {category}: {count:,} rows ({percentage:.1f}%)")

# Score histogram
print("\n📊 Score Histogram:")
score_bins = [0, 2, 4, 6, 8, 10]
bin_labels = ['0-2', '2-4', '4-6', '6-8', '8-10']
df['score_bin'] = pd.cut(df['camelbert_sentiment_score'], 
                         bins=score_bins, 
                         labels=bin_labels,
                         include_lowest=True)

bin_counts = df['score_bin'].value_counts().sort_index()
for bin_label, count in bin_counts.items():
    percentage = (count / len(df)) * 100
    print(f"  {bin_label}: {count:,} rows ({percentage:.1f}%)")

# %%
# Compare with existing rating if available
if 'rating' in df.columns:
    print("\n" + "="*80)
    print("🔄 COMPARISON WITH EXISTING RATINGS")
    print("="*80)
    
    # Convert existing rating to same scale if needed
    if df['rating'].max() <= 10:
        existing_rating = df['rating']
    else:
        # Normalize to 0-10 scale
        existing_rating = (df['rating'] / df['rating'].max()) * 10
    
    # Calculate correlation
    correlation = df['camelbert_sentiment_score'].corr(existing_rating)
    print(f"📐 Correlation with existing rating: {correlation:.3f}")
    
    # Difference analysis
    df['score_difference'] = df['camelbert_sentiment_score'] - existing_rating
    avg_diff = df['score_difference'].mean()
    abs_avg_diff = df['score_difference'].abs().mean()
    
    print(f"📊 Average difference: {avg_diff:.2f}")
    print(f"📊 Average absolute difference: {abs_avg_diff:.2f}")
    
    # Show examples with high differences
    print("\n🔍 Examples with largest differences:")
    
    # Top 5 where our score is higher
    print("\nTop 5 where CAMeL-Lab score is HIGHER than rating:")
    high_diff = df.nlargest(5, 'score_difference')
    for idx, row in high_diff.iterrows():
        print(f"  Score: {row['camelbert_sentiment_score']:.1f} vs "
              f"Rating: {row['rating']} "
              f"(Diff: {row['score_difference']:.1f})")
    
    # Top 5 where our score is lower
    print("\nTop 5 where CAMeL-Lab score is LOWER than rating:")
    low_diff = df.nsmallest(5, 'score_difference')
    for idx, row in low_diff.iterrows():
        print(f"  Score: {row['camelbert_sentiment_score']:.1f} vs "
              f"Rating: {row['rating']} "
              f"(Diff: {row['score_difference']:.1f})")

# %%
# Save results
output_path = '../data/tokenized/data_with_camelbert_scores.csv'
df.to_csv(output_path, index=False, encoding='utf-8-sig')

print(f"\n💾 Results saved to: {output_path}")
print(f"📁 File contains: {len(df):,} rows × {len(df.columns)} columns")

# Show new columns added
original_cols = pd.read_csv('../data/tokenized/data_cleaned_full_with_tokens.csv', nrows=0).columns
new_cols = [col for col in df.columns if col not in original_cols]
print(f"\n➕ New columns added ({len(new_cols)}):")
for col in new_cols:
    print(f"  - {col}")

# %%
# Display sample of results
print("\n" + "="*80)
print("👁️  SAMPLE OF RESULTS (First 10 rows)")
print("="*80)

sample_cols = ['combined_text', 'camelbert_sentiment_score', 'camelbert_sentiment_category']
if 'rating' in df.columns:
    sample_cols.insert(1, 'rating')

pd.set_option('display.max_colwidth', 100)
print(df[sample_cols].head(10).to_string())

# %%
# Visualization (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\n" + "="*80)
    print("📊 VISUALIZATIONS")
    print("="*80)
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Score distribution histogram
    axes[0, 0].hist(df['camelbert_sentiment_score'], bins=20, 
                   color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Score Distribution (0-10 scale)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Sentiment Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Category distribution
    category_order = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    category_counts = df['camelbert_sentiment_category'].value_counts()
    colors = ['#FF6B6B', '#FFA8A8', '#FFD166', '#06D6A0', '#118AB2']
    axes[0, 1].bar(category_order, [category_counts.get(cat, 0) for cat in category_order],
                   color=colors, edgecolor='black')
    axes[0, 1].set_title('Sentiment Categories', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Category')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Score boxplot by category
    category_data = [df[df['camelbert_sentiment_category'] == cat]['camelbert_sentiment_score']
                    for cat in category_order if cat in df['camelbert_sentiment_category'].unique()]
    axes[1, 0].boxplot(category_data, labels=[cat for cat in category_order 
                     if cat in df['camelbert_sentiment_category'].unique()])
    axes[1, 0].set_title('Score Distribution by Category', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Category')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Score vs Rating scatter (if available)
    if 'rating' in df.columns:
        axes[1, 1].scatter(df['rating'], df['camelbert_sentiment_score'], 
                          alpha=0.5, color='purple')
        axes[1, 1].set_title('CAMeL-Lab Score vs Existing Rating', 
                           fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Existing Rating')
        axes[1, 1].set_ylabel('CAMeL-Lab Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add correlation line
        z = np.polyfit(df['rating'], df['camelbert_sentiment_score'], 1)
        p = np.poly1d(z)
        axes[1, 1].plot(df['rating'], p(df['rating']), "r--", alpha=0.8)
        axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                       transform=axes[1, 1].transAxes, fontsize=10,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        axes[1, 1].text(0.5, 0.5, 'No rating data available\nfor comparison',
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=axes[1, 1].transAxes,
                       fontsize=12)
        axes[1, 1].set_title('Score Comparison', fontsize=14, fontweight='bold')
    
    plt.suptitle('CAMeL-Lab Sentiment Analysis Results', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()
    
except ImportError:
    print("Matplotlib not installed. Skipping visualizations.")
    print("To install: !pip install matplotlib seaborn")

# %%
# Performance summary
print("\n" + "="*80)
print("📋 PERFORMANCE SUMMARY")
print("="*80)
print(f"✅ Dataset size: {len(df):,} rows")
print(f"✅ New column added: camelbert_sentiment_score (0-10 scale)")
print(f"✅ Additional column: camelbert_sentiment_category")
print(f"✅ Average sentiment score: {df['camelbert_sentiment_score'].mean():.2f}/10")
print("="*80)
# Cleanup
gc.collect()
print("\n🧹 Memory cleanup complete")