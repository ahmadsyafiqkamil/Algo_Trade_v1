import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from libs.technical_analyzer import TechnicalAnalyzer
from libs.volume_profile_analyzer import VolumeProfileAnalyzer

class PumpRankingAnalyzer:
    def __init__(self):
        """Initialize PumpRankingAnalyzer with TechnicalAnalyzer and VolumeProfileAnalyzer"""
        self.technical_analyzer = TechnicalAnalyzer()
        self.volume_analyzer = VolumeProfileAnalyzer()
        
    def calculate_rank_score(self, df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive pump ranking score based on all indicators
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV and indicator data
            
        Returns:
            Dict containing ranking scores and analysis
        """
        # Initialize scores dictionary
        scores = {
            'trend': 0,
            'volume': 0,
            'patterns': 0,
            'volume_profile': 0,
            'support_resistance': 0,
            'momentum': 0,
            'total_score': 0
        }
        
        # Get the latest data point
        latest = df.iloc[-1]
        
        # 1. Trend Analysis (0-100)
        trend_score = latest['trend_strength']
        scores['trend'] = trend_score
        
        # 2. Volume Analysis (0-100)
        volume_score = 0
        if latest['volume_trend'] >= 2.0:
            volume_score = 100
        elif latest['volume_trend'] >= 1.5:
            volume_score = 80
        elif latest['volume_trend'] >= 1.2:
            volume_score = 60
        elif latest['volume_trend'] >= 1.0:
            volume_score = 40
        scores['volume'] = volume_score
        
        # 3. Pattern Analysis (0-100)
        pattern_score = 0
        if latest['hammer'] or latest['morning_star']:
            pattern_score = 100
        elif latest['engulfing'] == 1:  # Bullish engulfing
            pattern_score = 80
        elif latest['doji']:
            pattern_score = 40
        scores['patterns'] = pattern_score
        
        # 4. Volume Profile Analysis (0-100)
        vp_score = 0
        if latest['va_position'] < 0.2:  # Price below Value Area
            vp_score = 100
        elif latest['va_position'] < 0.4:
            vp_score = 80
        elif latest['va_position'] > 0.8:  # Price above Value Area
            vp_score = 40
        scores['volume_profile'] = vp_score
        
        # 5. Support/Resistance Analysis (0-100)
        sr_score = 0
        if latest['nearest_support'] is not None and latest['nearest_resistance'] is not None:
            price_range = latest['nearest_resistance'] - latest['nearest_support']
            current_price = latest['close']
            distance_to_support = (current_price - latest['nearest_support']) / price_range
            if distance_to_support < 0.2:  # Price near support
                sr_score = 100
            elif distance_to_support < 0.4:
                sr_score = 60
        scores['support_resistance'] = sr_score
        
        # 6. Momentum Analysis (0-100)
        momentum_score = 0
        if latest['macd'] > latest['macd_signal']:
            momentum_score = 80
            if latest['macd'] > 0:  # MACD above zero
                momentum_score = 100
        scores['momentum'] = momentum_score
        
        # Calculate total score with weights
        weights = {
            'trend': 0.25,          # 25%
            'volume': 0.20,         # 20%
            'patterns': 0.15,       # 15%
            'volume_profile': 0.15, # 15%
            'support_resistance': 0.15, # 15%
            'momentum': 0.10        # 10%
        }
        
        total_score = sum(scores[component] * weights[component] for component in weights)
        scores['total_score'] = total_score
        
        # Add ranking category
        if total_score >= 80:
            ranking = "HIGH"
        elif total_score >= 60:
            ranking = "MEDIUM"
        else:
            ranking = "LOW"
        scores['ranking'] = ranking
        
        return scores
    
    def analyze_symbol(self, df: pd.DataFrame) -> Dict:
        """
        Perform complete pump ranking analysis
        
        Args:
            df (pd.DataFrame): DataFrame with OHLCV data
            
        Returns:
            Dict containing ranking analysis and recommendations
        """
        # Calculate all indicators
        df = self.technical_analyzer.analyze_trend(df)
        df = self.technical_analyzer.detect_candle_patterns(df)
        df = self.technical_analyzer.analyze_volume(df)
        
        # Add volume profile analysis
        volume_profile = self.volume_analyzer.analyze(df)
        df['vwap'] = volume_profile['vwap']
        df['vpoc'] = volume_profile['vpoc']
        df['vah'] = volume_profile['vah']
        df['val'] = volume_profile['val']
        df['va_position'] = volume_profile['va_position']
        
        # Calculate ranking scores
        scores = self.calculate_rank_score(df)
        
        # Add analysis and recommendations
        latest = df.iloc[-1]
        analysis = {
            'scores': scores,
            'current_price': latest['close'],
            'timestamp': latest.name,
            'recommendations': self._generate_recommendations(scores, latest)
        }
        
        return analysis
    
    def _generate_recommendations(self, scores: Dict, latest: pd.Series) -> List[str]:
        """
        Generate trading recommendations based on scores
        
        Args:
            scores (Dict): Ranking scores
            latest (pd.Series): Latest data point
            
        Returns:
            List[str]: List of recommendations
        """
        recommendations = []
        
        # Overall recommendation
        if scores['ranking'] == "HIGH":
            recommendations.append("Strong pump potential detected")
        elif scores['ranking'] == "MEDIUM":
            recommendations.append("Moderate pump potential, monitor closely")
        else:
            recommendations.append("Low pump potential, wait for better conditions")
        
        # Specific recommendations based on components
        if scores['trend'] < 60:
            recommendations.append("Wait for stronger trend confirmation")
            
        if scores['volume'] < 60:
            recommendations.append("Volume needs to increase for stronger signal")
            
        if scores['patterns'] < 60:
            recommendations.append("Look for stronger bullish patterns")
            
        if scores['volume_profile'] < 60:
            recommendations.append("Price position relative to Value Area is not optimal")
            
        if scores['support_resistance'] < 60:
            recommendations.append("Price not in optimal support/resistance position")
            
        if scores['momentum'] < 60:
            recommendations.append("Momentum indicators need to strengthen")
        
        return recommendations
    
    def format_ranking_output(self, analysis: Dict) -> str:
        """
        Format ranking analysis for display
        
        Args:
            analysis (Dict): Analysis results
            
        Returns:
            str: Formatted output string
        """
        output = []
        output.append(f"Pump Ranking Analysis - {analysis['timestamp']}")
        output.append(f"Current Price: {analysis['current_price']:.8f}")
        output.append(f"Overall Ranking: {analysis['scores']['ranking']} ({analysis['scores']['total_score']:.1f}%)")
        output.append("\nComponent Scores:")
        output.append(f"Trend: {analysis['scores']['trend']:.1f}%")
        output.append(f"Volume: {analysis['scores']['volume']:.1f}%")
        output.append(f"Patterns: {analysis['scores']['patterns']:.1f}%")
        output.append(f"Volume Profile: {analysis['scores']['volume_profile']:.1f}%")
        output.append(f"Support/Resistance: {analysis['scores']['support_resistance']:.1f}%")
        output.append(f"Momentum: {analysis['scores']['momentum']:.1f}%")
        output.append("\nRecommendations:")
        for rec in analysis['recommendations']:
            output.append(f"- {rec}")
            
        return "\n".join(output) 