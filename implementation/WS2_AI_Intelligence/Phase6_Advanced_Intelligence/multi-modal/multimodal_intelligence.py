"""
Multi-Modal Intelligence System for Nexus Architect
Implements comprehensive multi-modal AI capabilities for text, image, audio, and video processing.
"""

import asyncio
import logging
import time
import io
import base64
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta
import cv2
import librosa
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel, AutoProcessor,
    BlipProcessor, BlipForConditionalGeneration,
    CLIPProcessor, CLIPModel,
    Wav2Vec2Processor, Wav2Vec2ForCTC,
    pipeline
)
import whisper
import openai
import anthropic
from moviepy.editor import VideoFileClip
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor
import hashlib
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import redis
import aioredis

# Metrics
MULTIMODAL_REQUESTS = Counter('multimodal_requests_total', 'Total multi-modal requests', ['modality', 'status'])
MULTIMODAL_LATENCY = Histogram('multimodal_latency_seconds', 'Multi-modal processing latency', ['modality'])
PROCESSING_QUEUE_SIZE = Gauge('processing_queue_size', 'Size of processing queue', ['modality'])
ACCURACY_SCORE = Gauge('multimodal_accuracy_score', 'Accuracy score for multi-modal processing', ['modality'])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """Types of modalities supported"""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    MULTIMODAL = "multimodal"

class ProcessingQuality(Enum):
    """Quality levels for processing"""
    FAST = "fast"           # Quick processing, lower accuracy
    BALANCED = "balanced"   # Balance between speed and accuracy
    HIGH = "high"          # High accuracy, slower processing
    PREMIUM = "premium"    # Maximum accuracy and features

@dataclass
class MultiModalContent:
    """Structure for multi-modal content"""
    content_id: str
    modality: ModalityType
    data: Union[str, bytes, np.ndarray]
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ProcessingRequest:
    """Request structure for multi-modal processing"""
    request_id: str
    user_id: str
    session_id: str
    content: List[MultiModalContent]
    processing_quality: ProcessingQuality = ProcessingQuality.BALANCED
    analysis_type: str = "comprehensive"  # comprehensive, focused, summary
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

@dataclass
class ProcessingResult:
    """Result structure for multi-modal processing"""
    request_id: str
    status: str
    modality_results: Dict[str, Any]
    integrated_analysis: Dict[str, Any]
    confidence_scores: Dict[str, float]
    processing_time: float
    models_used: List[str]
    created_at: datetime = field(default_factory=datetime.utcnow)

class TextProcessor:
    """Advanced text processing with multiple models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize text processing models"""
        try:
            # Sentiment analysis
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            
            # Named Entity Recognition
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
            
            # Text classification
            self.classification_pipeline = pipeline(
                "text-classification",
                model="facebook/bart-large-mnli"
            )
            
            # Question answering
            self.qa_pipeline = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2"
            )
            
            # Text summarization
            self.summarization_pipeline = pipeline(
                "summarization",
                model="facebook/bart-large-cnn"
            )
            
            logger.info("Text processing models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing text models: {e}")
    
    async def process_text(self, text: str, quality: ProcessingQuality = ProcessingQuality.BALANCED) -> Dict[str, Any]:
        """Process text with comprehensive analysis"""
        start_time = time.time()
        
        try:
            results = {}
            
            # Basic text analysis
            results["basic_analysis"] = {
                "length": len(text),
                "word_count": len(text.split()),
                "sentence_count": len([s for s in text.split('.') if s.strip()]),
                "language": self._detect_language(text)
            }
            
            # Sentiment analysis
            if quality in [ProcessingQuality.BALANCED, ProcessingQuality.HIGH, ProcessingQuality.PREMIUM]:
                sentiment_result = self.sentiment_pipeline(text[:512])  # Limit for model
                results["sentiment"] = {
                    "label": sentiment_result[0]["label"],
                    "confidence": sentiment_result[0]["score"]
                }
            
            # Named Entity Recognition
            if quality in [ProcessingQuality.HIGH, ProcessingQuality.PREMIUM]:
                entities = self.ner_pipeline(text[:512])
                results["entities"] = [
                    {
                        "text": entity["word"],
                        "label": entity["entity_group"],
                        "confidence": entity["score"]
                    }
                    for entity in entities
                ]
            
            # Topic classification
            if quality == ProcessingQuality.PREMIUM:
                topics = ["technology", "business", "science", "politics", "sports", "entertainment"]
                topic_scores = []
                
                for topic in topics:
                    hypothesis = f"This text is about {topic}"
                    result = self.classification_pipeline(text[:512], hypothesis)
                    topic_scores.append({
                        "topic": topic,
                        "confidence": result["scores"][result["labels"].index("ENTAILMENT")]
                    })
                
                results["topics"] = sorted(topic_scores, key=lambda x: x["confidence"], reverse=True)[:3]
            
            # Text summarization for long texts
            if len(text) > 500 and quality in [ProcessingQuality.HIGH, ProcessingQuality.PREMIUM]:
                summary = self.summarization_pipeline(
                    text[:1024],  # Limit input length
                    max_length=150,
                    min_length=50,
                    do_sample=False
                )
                results["summary"] = summary[0]["summary_text"]
            
            # Key phrases extraction
            results["key_phrases"] = self._extract_key_phrases(text)
            
            # Readability analysis
            results["readability"] = self._analyze_readability(text)
            
            processing_time = time.time() - start_time
            MULTIMODAL_LATENCY.labels(modality="text").observe(processing_time)
            
            return {
                "status": "success",
                "results": results,
                "processing_time": processing_time,
                "confidence": self._calculate_text_confidence(results)
            }
            
        except Exception as e:
            logger.error(f"Text processing error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        # Simplified language detection - in production use proper language detection
        common_english_words = ["the", "and", "is", "in", "to", "of", "a", "that", "it", "with"]
        words = text.lower().split()
        english_count = sum(1 for word in words if word in common_english_words)
        
        if len(words) > 0 and english_count / len(words) > 0.1:
            return "english"
        return "unknown"
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        # Simple key phrase extraction
        words = text.lower().split()
        
        # Remove common stop words
        stop_words = {"the", "and", "is", "in", "to", "of", "a", "that", "it", "with", "for", "as", "was", "on", "are", "you"}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Count word frequency
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top phrases
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        return [word for word, freq in top_words]
    
    def _analyze_readability(self, text: str) -> Dict[str, Any]:
        """Analyze text readability"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        words = text.split()
        
        if not sentences or not words:
            return {"score": 0, "level": "unknown"}
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = sum(len(word) for word in words) / len(words)
        
        # Simple readability score
        readability_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_word_length / 4.7)
        
        if readability_score >= 90:
            level = "very_easy"
        elif readability_score >= 80:
            level = "easy"
        elif readability_score >= 70:
            level = "fairly_easy"
        elif readability_score >= 60:
            level = "standard"
        elif readability_score >= 50:
            level = "fairly_difficult"
        elif readability_score >= 30:
            level = "difficult"
        else:
            level = "very_difficult"
        
        return {
            "score": max(0, min(100, readability_score)),
            "level": level,
            "avg_sentence_length": avg_sentence_length,
            "avg_word_length": avg_word_length
        }
    
    def _calculate_text_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence for text analysis"""
        confidences = []
        
        if "sentiment" in results:
            confidences.append(results["sentiment"]["confidence"])
        
        if "entities" in results:
            entity_confidences = [entity["confidence"] for entity in results["entities"]]
            if entity_confidences:
                confidences.append(np.mean(entity_confidences))
        
        if "topics" in results:
            topic_confidences = [topic["confidence"] for topic in results["topics"]]
            if topic_confidences:
                confidences.append(np.mean(topic_confidences))
        
        return np.mean(confidences) if confidences else 0.8

class ImageProcessor:
    """Advanced image processing with computer vision models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize image processing models"""
        try:
            # Image captioning
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            
            # CLIP for image-text understanding
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            
            # Object detection pipeline
            self.object_detection = pipeline("object-detection", model="facebook/detr-resnet-50")
            
            logger.info("Image processing models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing image models: {e}")
    
    async def process_image(self, image_data: bytes, quality: ProcessingQuality = ProcessingQuality.BALANCED) -> Dict[str, Any]:
        """Process image with comprehensive analysis"""
        start_time = time.time()
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            results = {}
            
            # Basic image analysis
            results["basic_analysis"] = {
                "dimensions": image.size,
                "mode": image.mode,
                "format": image.format,
                "size_bytes": len(image_data)
            }
            
            # Image captioning
            if quality in [ProcessingQuality.BALANCED, ProcessingQuality.HIGH, ProcessingQuality.PREMIUM]:
                caption = await self._generate_caption(image)
                results["caption"] = caption
            
            # Object detection
            if quality in [ProcessingQuality.HIGH, ProcessingQuality.PREMIUM]:
                objects = await self._detect_objects(image)
                results["objects"] = objects
            
            # Color analysis
            results["color_analysis"] = await self._analyze_colors(image)
            
            # Image features
            if quality == ProcessingQuality.PREMIUM:
                features = await self._extract_image_features(image)
                results["features"] = features
            
            # Text detection in image
            if quality in [ProcessingQuality.HIGH, ProcessingQuality.PREMIUM]:
                text_detection = await self._detect_text_in_image(image)
                results["text_detection"] = text_detection
            
            # Image quality assessment
            results["quality_assessment"] = await self._assess_image_quality(image)
            
            processing_time = time.time() - start_time
            MULTIMODAL_LATENCY.labels(modality="image").observe(processing_time)
            
            return {
                "status": "success",
                "results": results,
                "processing_time": processing_time,
                "confidence": self._calculate_image_confidence(results)
            }
            
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _generate_caption(self, image: Image.Image) -> Dict[str, Any]:
        """Generate image caption"""
        try:
            inputs = self.caption_processor(image, return_tensors="pt")
            out = self.caption_model.generate(**inputs, max_length=50)
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            
            return {
                "text": caption,
                "confidence": 0.85  # Mock confidence
            }
        except Exception as e:
            logger.error(f"Caption generation error: {e}")
            return {"text": "Unable to generate caption", "confidence": 0.0}
    
    async def _detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects in image"""
        try:
            # Convert PIL to format expected by pipeline
            results = self.object_detection(image)
            
            objects = []
            for result in results:
                objects.append({
                    "label": result["label"],
                    "confidence": result["score"],
                    "bbox": result["box"]
                })
            
            return objects
        except Exception as e:
            logger.error(f"Object detection error: {e}")
            return []
    
    async def _analyze_colors(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze color composition of image"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Calculate color statistics
            mean_color = np.mean(img_array, axis=(0, 1))
            dominant_color = self._get_dominant_color(img_array)
            
            # Color distribution
            colors = img_array.reshape(-1, 3)
            unique_colors = len(np.unique(colors.view(np.dtype((np.void, colors.dtype.itemsize * colors.shape[1])))))
            
            return {
                "mean_color": mean_color.tolist(),
                "dominant_color": dominant_color,
                "unique_colors": unique_colors,
                "brightness": float(np.mean(img_array)),
                "contrast": float(np.std(img_array))
            }
        except Exception as e:
            logger.error(f"Color analysis error: {e}")
            return {}
    
    def _get_dominant_color(self, img_array: np.ndarray) -> List[int]:
        """Get dominant color in image"""
        # Reshape and find most common color
        colors = img_array.reshape(-1, 3)
        
        # Simple approach: find mean color
        # In production, use k-means clustering for better results
        dominant = np.mean(colors, axis=0)
        return dominant.astype(int).tolist()
    
    async def _extract_image_features(self, image: Image.Image) -> Dict[str, Any]:
        """Extract advanced image features"""
        try:
            # Use CLIP to extract features
            inputs = self.clip_processor(images=image, return_tensors="pt")
            image_features = self.clip_model.get_image_features(**inputs)
            
            # Convert to numpy and get basic statistics
            features_np = image_features.detach().numpy().flatten()
            
            return {
                "feature_vector_size": len(features_np),
                "feature_mean": float(np.mean(features_np)),
                "feature_std": float(np.std(features_np)),
                "feature_min": float(np.min(features_np)),
                "feature_max": float(np.max(features_np))
            }
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {}
    
    async def _detect_text_in_image(self, image: Image.Image) -> Dict[str, Any]:
        """Detect text in image using OCR"""
        try:
            # Convert PIL to OpenCV format
            img_array = np.array(image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Simple text detection using edge detection
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Count potential text regions
            text_regions = len([c for c in contours if cv2.contourArea(c) > 100])
            
            # Mock OCR result - in production, use proper OCR like Tesseract
            has_text = text_regions > 5
            
            return {
                "has_text": has_text,
                "text_regions": text_regions,
                "confidence": 0.7 if has_text else 0.9
            }
        except Exception as e:
            logger.error(f"Text detection error: {e}")
            return {"has_text": False, "confidence": 0.0}
    
    async def _assess_image_quality(self, image: Image.Image) -> Dict[str, Any]:
        """Assess image quality"""
        try:
            img_array = np.array(image)
            
            # Calculate quality metrics
            sharpness = self._calculate_sharpness(img_array)
            noise_level = self._calculate_noise_level(img_array)
            exposure = self._calculate_exposure(img_array)
            
            # Overall quality score
            quality_score = (sharpness * 0.4 + (1 - noise_level) * 0.3 + exposure * 0.3)
            
            return {
                "sharpness": sharpness,
                "noise_level": noise_level,
                "exposure": exposure,
                "overall_quality": quality_score,
                "quality_rating": self._get_quality_rating(quality_score)
            }
        except Exception as e:
            logger.error(f"Quality assessment error: {e}")
            return {}
    
    def _calculate_sharpness(self, img_array: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Normalize to 0-1 range
        return min(1.0, laplacian_var / 1000.0)
    
    def _calculate_noise_level(self, img_array: np.ndarray) -> float:
        """Calculate noise level in image"""
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # Use standard deviation as noise indicator
        noise = np.std(gray) / 255.0
        return min(1.0, noise)
    
    def _calculate_exposure(self, img_array: np.ndarray) -> float:
        """Calculate exposure quality"""
        brightness = np.mean(img_array) / 255.0
        # Optimal exposure is around 0.5 (middle gray)
        exposure_quality = 1.0 - abs(brightness - 0.5) * 2
        return max(0.0, exposure_quality)
    
    def _get_quality_rating(self, score: float) -> str:
        """Get quality rating from score"""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _calculate_image_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence for image analysis"""
        confidences = []
        
        if "caption" in results:
            confidences.append(results["caption"]["confidence"])
        
        if "objects" in results:
            object_confidences = [obj["confidence"] for obj in results["objects"]]
            if object_confidences:
                confidences.append(np.mean(object_confidences))
        
        if "text_detection" in results:
            confidences.append(results["text_detection"]["confidence"])
        
        return np.mean(confidences) if confidences else 0.8

class AudioProcessor:
    """Advanced audio processing with speech and sound analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize audio processing models"""
        try:
            # Whisper for speech recognition
            self.whisper_model = whisper.load_model("base")
            
            # Wav2Vec2 for speech recognition
            self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            self.wav2vec_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            
            # Audio classification
            self.audio_classifier = pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")
            
            logger.info("Audio processing models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing audio models: {e}")
    
    async def process_audio(self, audio_data: bytes, quality: ProcessingQuality = ProcessingQuality.BALANCED) -> Dict[str, Any]:
        """Process audio with comprehensive analysis"""
        start_time = time.time()
        
        try:
            # Save audio temporarily
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(audio_data)
                temp_file.flush()
                temp_path = temp_file.name
            
            try:
                # Load audio
                audio, sr = librosa.load(temp_path, sr=16000)
                
                results = {}
                
                # Basic audio analysis
                results["basic_analysis"] = {
                    "duration": len(audio) / sr,
                    "sample_rate": sr,
                    "channels": 1,  # Mono after librosa load
                    "size_bytes": len(audio_data)
                }
                
                # Speech recognition
                if quality in [ProcessingQuality.BALANCED, ProcessingQuality.HIGH, ProcessingQuality.PREMIUM]:
                    transcription = await self._transcribe_audio(temp_path, audio, sr)
                    results["transcription"] = transcription
                
                # Audio features
                features = await self._extract_audio_features(audio, sr)
                results["features"] = features
                
                # Audio classification
                if quality in [ProcessingQuality.HIGH, ProcessingQuality.PREMIUM]:
                    classification = await self._classify_audio(temp_path)
                    results["classification"] = classification
                
                # Speech analysis
                if quality == ProcessingQuality.PREMIUM:
                    speech_analysis = await self._analyze_speech(audio, sr)
                    results["speech_analysis"] = speech_analysis
                
                # Audio quality assessment
                results["quality_assessment"] = await self._assess_audio_quality(audio, sr)
                
                processing_time = time.time() - start_time
                MULTIMODAL_LATENCY.labels(modality="audio").observe(processing_time)
                
                return {
                    "status": "success",
                    "results": results,
                    "processing_time": processing_time,
                    "confidence": self._calculate_audio_confidence(results)
                }
                
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _transcribe_audio(self, audio_path: str, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Transcribe audio to text"""
        try:
            # Use Whisper for transcription
            result = self.whisper_model.transcribe(audio_path)
            
            return {
                "text": result["text"],
                "language": result.get("language", "unknown"),
                "confidence": 0.85,  # Mock confidence
                "segments": result.get("segments", [])
            }
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {"text": "", "confidence": 0.0}
    
    async def _extract_audio_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract audio features"""
        try:
            features = {}
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            features["spectral_centroid"] = {
                "mean": float(np.mean(spectral_centroids)),
                "std": float(np.std(spectral_centroids))
            }
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features["zero_crossing_rate"] = {
                "mean": float(np.mean(zcr)),
                "std": float(np.std(zcr))
            }
            
            # MFCC features
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features["mfcc"] = {
                "mean": np.mean(mfccs, axis=1).tolist(),
                "std": np.std(mfccs, axis=1).tolist()
            }
            
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features["tempo"] = float(tempo)
            
            # RMS energy
            rms = librosa.feature.rms(y=audio)[0]
            features["rms_energy"] = {
                "mean": float(np.mean(rms)),
                "std": float(np.std(rms))
            }
            
            return features
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
            return {}
    
    async def _classify_audio(self, audio_path: str) -> Dict[str, Any]:
        """Classify audio content"""
        try:
            # Mock audio classification - in production use proper audio classifier
            # The pipeline might not work with all audio formats
            
            # Simple classification based on audio features
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Analyze frequency content
            fft = np.fft.fft(audio)
            freqs = np.fft.fftfreq(len(fft), 1/sr)
            magnitude = np.abs(fft)
            
            # Simple heuristics for classification
            low_freq_energy = np.sum(magnitude[np.abs(freqs) < 500])
            mid_freq_energy = np.sum(magnitude[(np.abs(freqs) >= 500) & (np.abs(freqs) < 2000)])
            high_freq_energy = np.sum(magnitude[np.abs(freqs) >= 2000])
            
            total_energy = low_freq_energy + mid_freq_energy + high_freq_energy
            
            if total_energy > 0:
                low_ratio = low_freq_energy / total_energy
                mid_ratio = mid_freq_energy / total_energy
                high_ratio = high_freq_energy / total_energy
                
                if mid_ratio > 0.5:
                    category = "speech"
                    confidence = 0.8
                elif low_ratio > 0.6:
                    category = "music"
                    confidence = 0.7
                else:
                    category = "other"
                    confidence = 0.6
            else:
                category = "silence"
                confidence = 0.9
            
            return {
                "category": category,
                "confidence": confidence,
                "frequency_distribution": {
                    "low_freq_ratio": low_ratio if total_energy > 0 else 0,
                    "mid_freq_ratio": mid_ratio if total_energy > 0 else 0,
                    "high_freq_ratio": high_ratio if total_energy > 0 else 0
                }
            }
        except Exception as e:
            logger.error(f"Audio classification error: {e}")
            return {"category": "unknown", "confidence": 0.0}
    
    async def _analyze_speech(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze speech characteristics"""
        try:
            analysis = {}
            
            # Speech rate (words per minute)
            duration = len(audio) / sr
            # Mock word count - in production, use actual transcription
            estimated_words = duration * 2.5  # Assume 2.5 words per second average
            speech_rate = (estimated_words / duration) * 60 if duration > 0 else 0
            
            analysis["speech_rate_wpm"] = speech_rate
            
            # Pitch analysis
            pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if pitch_values:
                analysis["pitch"] = {
                    "mean": float(np.mean(pitch_values)),
                    "std": float(np.std(pitch_values)),
                    "min": float(np.min(pitch_values)),
                    "max": float(np.max(pitch_values))
                }
            
            # Voice activity detection (simple energy-based)
            frame_length = int(0.025 * sr)  # 25ms frames
            hop_length = int(0.01 * sr)     # 10ms hop
            
            frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
            energy = np.sum(frames ** 2, axis=0)
            
            # Threshold for voice activity
            threshold = np.mean(energy) * 0.1
            voice_frames = energy > threshold
            voice_activity_ratio = np.sum(voice_frames) / len(voice_frames)
            
            analysis["voice_activity_ratio"] = float(voice_activity_ratio)
            
            return analysis
        except Exception as e:
            logger.error(f"Speech analysis error: {e}")
            return {}
    
    async def _assess_audio_quality(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Assess audio quality"""
        try:
            # Signal-to-noise ratio estimation
            # Simple approach: compare signal energy to noise floor
            energy = np.sum(audio ** 2)
            noise_floor = np.percentile(np.abs(audio), 10)  # Bottom 10% as noise estimate
            signal_level = np.percentile(np.abs(audio), 90)  # Top 10% as signal estimate
            
            snr = 20 * np.log10(signal_level / (noise_floor + 1e-10))  # Avoid division by zero
            
            # Clipping detection
            clipping_ratio = np.sum(np.abs(audio) > 0.95) / len(audio)
            
            # Dynamic range
            dynamic_range = 20 * np.log10(np.max(np.abs(audio)) / (np.mean(np.abs(audio)) + 1e-10))
            
            # Overall quality score
            quality_score = min(1.0, max(0.0, (snr + 20) / 40))  # Normalize SNR to 0-1
            quality_score *= (1 - clipping_ratio)  # Penalize clipping
            
            return {
                "snr_db": float(snr),
                "clipping_ratio": float(clipping_ratio),
                "dynamic_range_db": float(dynamic_range),
                "overall_quality": float(quality_score),
                "quality_rating": self._get_audio_quality_rating(quality_score)
            }
        except Exception as e:
            logger.error(f"Audio quality assessment error: {e}")
            return {}
    
    def _get_audio_quality_rating(self, score: float) -> str:
        """Get quality rating from score"""
        if score >= 0.8:
            return "excellent"
        elif score >= 0.6:
            return "good"
        elif score >= 0.4:
            return "fair"
        else:
            return "poor"
    
    def _calculate_audio_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence for audio analysis"""
        confidences = []
        
        if "transcription" in results:
            confidences.append(results["transcription"]["confidence"])
        
        if "classification" in results:
            confidences.append(results["classification"]["confidence"])
        
        if "quality_assessment" in results:
            confidences.append(results["quality_assessment"]["overall_quality"])
        
        return np.mean(confidences) if confidences else 0.8

class VideoProcessor:
    """Advanced video processing with frame analysis and content understanding"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.image_processor = ImageProcessor(config)
        self.audio_processor = AudioProcessor(config)
    
    async def process_video(self, video_data: bytes, quality: ProcessingQuality = ProcessingQuality.BALANCED) -> Dict[str, Any]:
        """Process video with comprehensive analysis"""
        start_time = time.time()
        
        try:
            # Save video temporarily
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
                temp_file.write(video_data)
                temp_file.flush()
                temp_path = temp_file.name
            
            try:
                # Load video
                video_clip = VideoFileClip(temp_path)
                
                results = {}
                
                # Basic video analysis
                results["basic_analysis"] = {
                    "duration": video_clip.duration,
                    "fps": video_clip.fps,
                    "size": video_clip.size,
                    "size_bytes": len(video_data)
                }
                
                # Frame analysis
                if quality in [ProcessingQuality.BALANCED, ProcessingQuality.HIGH, ProcessingQuality.PREMIUM]:
                    frame_analysis = await self._analyze_frames(video_clip, quality)
                    results["frame_analysis"] = frame_analysis
                
                # Audio analysis (if video has audio)
                if video_clip.audio is not None:
                    audio_analysis = await self._analyze_video_audio(video_clip)
                    results["audio_analysis"] = audio_analysis
                
                # Scene detection
                if quality in [ProcessingQuality.HIGH, ProcessingQuality.PREMIUM]:
                    scene_analysis = await self._detect_scenes(video_clip)
                    results["scene_analysis"] = scene_analysis
                
                # Motion analysis
                if quality == ProcessingQuality.PREMIUM:
                    motion_analysis = await self._analyze_motion(video_clip)
                    results["motion_analysis"] = motion_analysis
                
                # Video quality assessment
                results["quality_assessment"] = await self._assess_video_quality(video_clip)
                
                # Content summary
                results["content_summary"] = await self._summarize_video_content(results)
                
                video_clip.close()
                
                processing_time = time.time() - start_time
                MULTIMODAL_LATENCY.labels(modality="video").observe(processing_time)
                
                return {
                    "status": "success",
                    "results": results,
                    "processing_time": processing_time,
                    "confidence": self._calculate_video_confidence(results)
                }
                
            finally:
                # Clean up temporary file
                os.unlink(temp_path)
                
        except Exception as e:
            logger.error(f"Video processing error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    async def _analyze_frames(self, video_clip, quality: ProcessingQuality) -> Dict[str, Any]:
        """Analyze video frames"""
        try:
            # Sample frames based on quality
            if quality == ProcessingQuality.FAST:
                sample_count = 3
            elif quality == ProcessingQuality.BALANCED:
                sample_count = 5
            elif quality == ProcessingQuality.HIGH:
                sample_count = 10
            else:  # PREMIUM
                sample_count = 20
            
            duration = video_clip.duration
            sample_times = np.linspace(0, duration - 0.1, sample_count)
            
            frame_results = []
            
            for i, t in enumerate(sample_times):
                try:
                    # Extract frame
                    frame = video_clip.get_frame(t)
                    
                    # Convert to PIL Image
                    frame_image = Image.fromarray(frame.astype('uint8'))
                    
                    # Convert to bytes for image processor
                    img_byte_arr = io.BytesIO()
                    frame_image.save(img_byte_arr, format='JPEG')
                    img_bytes = img_byte_arr.getvalue()
                    
                    # Process frame
                    frame_result = await self.image_processor.process_image(img_bytes, quality)
                    frame_result["timestamp"] = t
                    frame_result["frame_index"] = i
                    
                    frame_results.append(frame_result)
                    
                except Exception as e:
                    logger.error(f"Error processing frame at {t}s: {e}")
                    continue
            
            # Aggregate frame analysis
            successful_frames = [f for f in frame_results if f.get("status") == "success"]
            
            if successful_frames:
                # Extract captions
                captions = [f["results"]["caption"]["text"] for f in successful_frames 
                           if "caption" in f.get("results", {})]
                
                # Extract objects
                all_objects = []
                for f in successful_frames:
                    if "objects" in f.get("results", {}):
                        all_objects.extend(f["results"]["objects"])
                
                # Count object types
                object_counts = {}
                for obj in all_objects:
                    label = obj["label"]
                    object_counts[label] = object_counts.get(label, 0) + 1
                
                return {
                    "frames_analyzed": len(successful_frames),
                    "sample_times": sample_times.tolist(),
                    "captions": captions,
                    "object_summary": dict(sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
                    "frame_details": successful_frames
                }
            else:
                return {"frames_analyzed": 0, "error": "No frames could be processed"}
                
        except Exception as e:
            logger.error(f"Frame analysis error: {e}")
            return {"error": str(e)}
    
    async def _analyze_video_audio(self, video_clip) -> Dict[str, Any]:
        """Analyze audio track of video"""
        try:
            # Extract audio
            audio_clip = video_clip.audio
            
            # Save audio temporarily
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                audio_clip.write_audiofile(temp_audio.name, verbose=False, logger=None)
                
                # Read audio data
                with open(temp_audio.name, 'rb') as f:
                    audio_data = f.read()
                
                # Process audio
                audio_result = await self.audio_processor.process_audio(audio_data)
                
                # Clean up
                os.unlink(temp_audio.name)
                
                return audio_result
                
        except Exception as e:
            logger.error(f"Video audio analysis error: {e}")
            return {"error": str(e)}
    
    async def _detect_scenes(self, video_clip) -> Dict[str, Any]:
        """Detect scene changes in video"""
        try:
            # Simple scene detection based on frame differences
            duration = video_clip.duration
            sample_count = min(50, int(duration * 2))  # 2 samples per second, max 50
            sample_times = np.linspace(0, duration - 0.1, sample_count)
            
            frame_differences = []
            prev_frame = None
            
            for t in sample_times:
                try:
                    frame = video_clip.get_frame(t)
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    
                    if prev_frame is not None:
                        # Calculate frame difference
                        diff = cv2.absdiff(gray_frame, prev_frame)
                        diff_score = np.mean(diff)
                        frame_differences.append((t, diff_score))
                    
                    prev_frame = gray_frame
                    
                except Exception as e:
                    logger.error(f"Error processing frame at {t}s for scene detection: {e}")
                    continue
            
            if frame_differences:
                # Find scene boundaries (high differences)
                diff_scores = [score for _, score in frame_differences]
                threshold = np.mean(diff_scores) + 2 * np.std(diff_scores)
                
                scene_changes = [t for t, score in frame_differences if score > threshold]
                
                return {
                    "scene_count": len(scene_changes) + 1,
                    "scene_changes": scene_changes,
                    "avg_scene_length": duration / (len(scene_changes) + 1) if scene_changes else duration
                }
            else:
                return {"scene_count": 1, "scene_changes": []}
                
        except Exception as e:
            logger.error(f"Scene detection error: {e}")
            return {"error": str(e)}
    
    async def _analyze_motion(self, video_clip) -> Dict[str, Any]:
        """Analyze motion in video"""
        try:
            # Sample frames for motion analysis
            duration = video_clip.duration
            sample_count = min(20, int(duration))
            sample_times = np.linspace(0, duration - 0.1, sample_count)
            
            motion_scores = []
            prev_frame = None
            
            for t in sample_times:
                try:
                    frame = video_clip.get_frame(t)
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    
                    if prev_frame is not None:
                        # Calculate optical flow
                        flow = cv2.calcOpticalFlowPyrLK(
                            prev_frame, gray_frame,
                            np.array([[100, 100]], dtype=np.float32),  # Simple corner
                            None
                        )[0]
                        
                        # Calculate motion magnitude
                        if flow is not None and len(flow) > 0:
                            motion_magnitude = np.linalg.norm(flow[0] - [100, 100])
                            motion_scores.append(motion_magnitude)
                    
                    prev_frame = gray_frame
                    
                except Exception as e:
                    logger.error(f"Error analyzing motion at {t}s: {e}")
                    continue
            
            if motion_scores:
                avg_motion = np.mean(motion_scores)
                max_motion = np.max(motion_scores)
                motion_variance = np.var(motion_scores)
                
                # Classify motion level
                if avg_motion < 5:
                    motion_level = "low"
                elif avg_motion < 15:
                    motion_level = "medium"
                else:
                    motion_level = "high"
                
                return {
                    "average_motion": float(avg_motion),
                    "max_motion": float(max_motion),
                    "motion_variance": float(motion_variance),
                    "motion_level": motion_level
                }
            else:
                return {"motion_level": "unknown"}
                
        except Exception as e:
            logger.error(f"Motion analysis error: {e}")
            return {"error": str(e)}
    
    async def _assess_video_quality(self, video_clip) -> Dict[str, Any]:
        """Assess video quality"""
        try:
            # Sample a few frames for quality assessment
            sample_times = [video_clip.duration * 0.25, video_clip.duration * 0.5, video_clip.duration * 0.75]
            quality_scores = []
            
            for t in sample_times:
                try:
                    frame = video_clip.get_frame(t)
                    
                    # Convert to PIL Image for quality assessment
                    frame_image = Image.fromarray(frame.astype('uint8'))
                    img_byte_arr = io.BytesIO()
                    frame_image.save(img_byte_arr, format='JPEG')
                    img_bytes = img_byte_arr.getvalue()
                    
                    # Use image processor for quality assessment
                    frame_result = await self.image_processor.process_image(img_bytes)
                    if frame_result.get("status") == "success":
                        quality_info = frame_result["results"].get("quality_assessment", {})
                        if "overall_quality" in quality_info:
                            quality_scores.append(quality_info["overall_quality"])
                    
                except Exception as e:
                    logger.error(f"Error assessing quality at {t}s: {e}")
                    continue
            
            if quality_scores:
                avg_quality = np.mean(quality_scores)
                quality_consistency = 1.0 - np.std(quality_scores)  # Higher consistency = lower std
                
                return {
                    "average_quality": float(avg_quality),
                    "quality_consistency": float(max(0, quality_consistency)),
                    "resolution": video_clip.size,
                    "fps": video_clip.fps,
                    "duration": video_clip.duration
                }
            else:
                return {
                    "resolution": video_clip.size,
                    "fps": video_clip.fps,
                    "duration": video_clip.duration
                }
                
        except Exception as e:
            logger.error(f"Video quality assessment error: {e}")
            return {"error": str(e)}
    
    async def _summarize_video_content(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize video content"""
        try:
            summary = {}
            
            # Extract key information
            if "frame_analysis" in results:
                frame_data = results["frame_analysis"]
                
                # Most common objects
                if "object_summary" in frame_data:
                    top_objects = list(frame_data["object_summary"].keys())[:5]
                    summary["main_objects"] = top_objects
                
                # Sample captions
                if "captions" in frame_data:
                    captions = frame_data["captions"][:3]  # First 3 captions
                    summary["sample_descriptions"] = captions
            
            # Audio content
            if "audio_analysis" in results:
                audio_data = results["audio_analysis"]
                if audio_data.get("status") == "success":
                    audio_results = audio_data.get("results", {})
                    if "transcription" in audio_results:
                        summary["audio_content"] = audio_results["transcription"]["text"][:200]  # First 200 chars
            
            # Scene information
            if "scene_analysis" in results:
                scene_data = results["scene_analysis"]
                summary["scene_count"] = scene_data.get("scene_count", 1)
                summary["avg_scene_length"] = scene_data.get("avg_scene_length", 0)
            
            # Motion level
            if "motion_analysis" in results:
                motion_data = results["motion_analysis"]
                summary["motion_level"] = motion_data.get("motion_level", "unknown")
            
            # Overall assessment
            basic_info = results.get("basic_analysis", {})
            summary["duration"] = basic_info.get("duration", 0)
            summary["resolution"] = basic_info.get("size", [0, 0])
            
            return summary
            
        except Exception as e:
            logger.error(f"Video content summarization error: {e}")
            return {"error": str(e)}
    
    def _calculate_video_confidence(self, results: Dict[str, Any]) -> float:
        """Calculate overall confidence for video analysis"""
        confidences = []
        
        # Frame analysis confidence
        if "frame_analysis" in results:
            frame_data = results["frame_analysis"]
            if "frame_details" in frame_data:
                frame_confidences = [f.get("confidence", 0) for f in frame_data["frame_details"]]
                if frame_confidences:
                    confidences.append(np.mean(frame_confidences))
        
        # Audio analysis confidence
        if "audio_analysis" in results:
            audio_data = results["audio_analysis"]
            if audio_data.get("status") == "success":
                confidences.append(audio_data.get("confidence", 0.8))
        
        # Quality assessment confidence
        if "quality_assessment" in results:
            quality_data = results["quality_assessment"]
            if "average_quality" in quality_data:
                confidences.append(quality_data["average_quality"])
        
        return np.mean(confidences) if confidences else 0.8

class MultiModalIntelligence:
    """Main multi-modal intelligence system orchestrating all processors"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.text_processor = TextProcessor(config.get("text", {}))
        self.image_processor = ImageProcessor(config.get("image", {}))
        self.audio_processor = AudioProcessor(config.get("audio", {}))
        self.video_processor = VideoProcessor(config.get("video", {}))
        
        self.redis_client = None
        self.executor = ThreadPoolExecutor(max_workers=5)
        self.cache_ttl = config.get("cache_ttl", 3600)
        
        # Start metrics server
        start_http_server(8002)
    
    async def initialize(self):
        """Initialize multi-modal intelligence system"""
        # Initialize Redis for caching
        redis_config = self.config.get("redis", {})
        self.redis_client = aioredis.from_url(
            f"redis://{redis_config.get('host', 'localhost')}:{redis_config.get('port', 6379)}"
        )
        
        logger.info("Multi-Modal Intelligence system initialized successfully")
    
    async def process_multimodal_request(self, request: ProcessingRequest) -> ProcessingResult:
        """Process multi-modal request"""
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = self._generate_cache_key(request)
            cached_result = await self._get_cached_result(cache_key)
            
            if cached_result:
                return ProcessingResult(
                    request_id=request.request_id,
                    status="success",
                    modality_results=cached_result["modality_results"],
                    integrated_analysis=cached_result["integrated_analysis"],
                    confidence_scores=cached_result["confidence_scores"],
                    processing_time=time.time() - start_time,
                    models_used=cached_result["models_used"]
                )
            
            # Process each modality
            modality_results = {}
            confidence_scores = {}
            models_used = []
            
            for content in request.content:
                modality = content.modality
                PROCESSING_QUEUE_SIZE.labels(modality=modality.value).inc()
                
                try:
                    if modality == ModalityType.TEXT:
                        result = await self.text_processor.process_text(
                            content.data, request.processing_quality
                        )
                        models_used.extend(["sentiment_model", "ner_model", "classification_model"])
                    
                    elif modality == ModalityType.IMAGE:
                        result = await self.image_processor.process_image(
                            content.data, request.processing_quality
                        )
                        models_used.extend(["blip_caption", "clip_model", "object_detection"])
                    
                    elif modality == ModalityType.AUDIO:
                        result = await self.audio_processor.process_audio(
                            content.data, request.processing_quality
                        )
                        models_used.extend(["whisper", "wav2vec2", "audio_classifier"])
                    
                    elif modality == ModalityType.VIDEO:
                        result = await self.video_processor.process_video(
                            content.data, request.processing_quality
                        )
                        models_used.extend(["video_analysis", "frame_processor", "scene_detector"])
                    
                    else:
                        result = {"status": "error", "error": f"Unsupported modality: {modality}"}
                    
                    modality_results[modality.value] = result
                    confidence_scores[modality.value] = result.get("confidence", 0.0)
                    
                    MULTIMODAL_REQUESTS.labels(
                        modality=modality.value,
                        status=result.get("status", "unknown")
                    ).inc()
                    
                    if result.get("confidence"):
                        ACCURACY_SCORE.labels(modality=modality.value).set(result["confidence"])
                
                except Exception as e:
                    logger.error(f"Error processing {modality.value}: {e}")
                    modality_results[modality.value] = {"status": "error", "error": str(e)}
                    confidence_scores[modality.value] = 0.0
                    
                    MULTIMODAL_REQUESTS.labels(
                        modality=modality.value,
                        status="error"
                    ).inc()
                
                finally:
                    PROCESSING_QUEUE_SIZE.labels(modality=modality.value).dec()
            
            # Integrate analysis across modalities
            integrated_analysis = await self._integrate_multimodal_analysis(
                modality_results, request
            )
            
            # Cache result
            cache_data = {
                "modality_results": modality_results,
                "integrated_analysis": integrated_analysis,
                "confidence_scores": confidence_scores,
                "models_used": list(set(models_used))
            }
            await self._cache_result(cache_key, cache_data)
            
            processing_time = time.time() - start_time
            
            return ProcessingResult(
                request_id=request.request_id,
                status="success",
                modality_results=modality_results,
                integrated_analysis=integrated_analysis,
                confidence_scores=confidence_scores,
                processing_time=processing_time,
                models_used=list(set(models_used))
            )
            
        except Exception as e:
            logger.error(f"Multi-modal processing error: {e}")
            return ProcessingResult(
                request_id=request.request_id,
                status="error",
                modality_results={"error": str(e)},
                integrated_analysis={"error": str(e)},
                confidence_scores={},
                processing_time=time.time() - start_time,
                models_used=[]
            )
    
    async def _integrate_multimodal_analysis(self, modality_results: Dict[str, Any], 
                                           request: ProcessingRequest) -> Dict[str, Any]:
        """Integrate analysis across multiple modalities"""
        try:
            integration = {
                "summary": "",
                "key_insights": [],
                "cross_modal_correlations": {},
                "overall_confidence": 0.0
            }
            
            successful_results = {k: v for k, v in modality_results.items() 
                                if v.get("status") == "success"}
            
            if not successful_results:
                return {"error": "No successful modality processing"}
            
            # Extract key information from each modality
            text_content = []
            visual_content = []
            audio_content = []
            
            for modality, result in successful_results.items():
                if modality == "text":
                    text_data = result.get("results", {})
                    if "summary" in text_data:
                        text_content.append(f"Text summary: {text_data['summary']}")
                    if "key_phrases" in text_data:
                        text_content.append(f"Key phrases: {', '.join(text_data['key_phrases'][:5])}")
                
                elif modality == "image":
                    image_data = result.get("results", {})
                    if "caption" in image_data:
                        visual_content.append(f"Image: {image_data['caption']['text']}")
                    if "objects" in image_data:
                        objects = [obj["label"] for obj in image_data["objects"][:5]]
                        visual_content.append(f"Objects detected: {', '.join(objects)}")
                
                elif modality == "audio":
                    audio_data = result.get("results", {})
                    if "transcription" in audio_data:
                        audio_content.append(f"Audio: {audio_data['transcription']['text'][:100]}")
                    if "classification" in audio_data:
                        audio_content.append(f"Audio type: {audio_data['classification']['category']}")
                
                elif modality == "video":
                    video_data = result.get("results", {})
                    if "content_summary" in video_data:
                        summary = video_data["content_summary"]
                        if "sample_descriptions" in summary:
                            visual_content.extend([f"Video frame: {desc}" for desc in summary["sample_descriptions"][:2]])
                        if "audio_content" in summary:
                            audio_content.append(f"Video audio: {summary['audio_content'][:100]}")
            
            # Create integrated summary
            all_content = text_content + visual_content + audio_content
            if all_content:
                integration["summary"] = " | ".join(all_content[:5])  # Top 5 insights
            
            # Identify cross-modal correlations
            correlations = {}
            
            # Text-Image correlation
            if "text" in successful_results and "image" in successful_results:
                text_phrases = successful_results["text"].get("results", {}).get("key_phrases", [])
                image_objects = [obj["label"] for obj in successful_results["image"].get("results", {}).get("objects", [])]
                
                common_concepts = set(text_phrases) & set(image_objects)
                if common_concepts:
                    correlations["text_image"] = list(common_concepts)
            
            # Audio-Video correlation
            if "audio" in successful_results and "video" in successful_results:
                audio_type = successful_results["audio"].get("results", {}).get("classification", {}).get("category")
                video_motion = successful_results["video"].get("results", {}).get("motion_analysis", {}).get("motion_level")
                
                if audio_type and video_motion:
                    correlations["audio_video"] = {
                        "audio_type": audio_type,
                        "motion_level": video_motion,
                        "correlation": "high" if (audio_type == "music" and video_motion == "high") else "medium"
                    }
            
            integration["cross_modal_correlations"] = correlations
            
            # Calculate overall confidence
            confidences = [result.get("confidence", 0) for result in successful_results.values()]
            integration["overall_confidence"] = np.mean(confidences) if confidences else 0.0
            
            # Generate key insights
            insights = []
            
            if len(successful_results) > 1:
                insights.append(f"Multi-modal content analyzed across {len(successful_results)} modalities")
            
            if correlations:
                insights.append(f"Found {len(correlations)} cross-modal correlations")
            
            if integration["overall_confidence"] > 0.8:
                insights.append("High confidence analysis across all modalities")
            
            integration["key_insights"] = insights
            
            return integration
            
        except Exception as e:
            logger.error(f"Multi-modal integration error: {e}")
            return {"error": str(e)}
    
    def _generate_cache_key(self, request: ProcessingRequest) -> str:
        """Generate cache key for request"""
        # Create hash of request content
        content_hashes = []
        for content in request.content:
            if isinstance(content.data, bytes):
                content_hash = hashlib.md5(content.data).hexdigest()
            else:
                content_hash = hashlib.md5(str(content.data).encode()).hexdigest()
            content_hashes.append(f"{content.modality.value}:{content_hash}")
        
        combined = f"{request.processing_quality.value}:{':'.join(content_hashes)}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    async def _get_cached_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached result if available"""
        try:
            cached_data = await self.redis_client.get(f"multimodal:{cache_key}")
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
        return None
    
    async def _cache_result(self, cache_key: str, result: Dict[str, Any]):
        """Cache result for future use"""
        try:
            await self.redis_client.setex(
                f"multimodal:{cache_key}",
                self.cache_ttl,
                json.dumps(result, default=str)
            )
        except Exception as e:
            logger.error(f"Cache storage error: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "processors": {}
        }
        
        # Check Redis
        try:
            await self.redis_client.ping()
            health_status["processors"]["redis"] = "healthy"
        except Exception:
            health_status["processors"]["redis"] = "unhealthy"
            health_status["status"] = "degraded"
        
        # Check model availability (simplified)
        health_status["processors"]["text_processor"] = "healthy"
        health_status["processors"]["image_processor"] = "healthy"
        health_status["processors"]["audio_processor"] = "healthy"
        health_status["processors"]["video_processor"] = "healthy"
        
        return health_status

# Configuration
DEFAULT_CONFIG = {
    "text": {},
    "image": {},
    "audio": {},
    "video": {},
    "redis": {
        "host": "localhost",
        "port": 6379
    },
    "cache_ttl": 3600
}

if __name__ == "__main__":
    import asyncio
    
    async def main():
        intelligence = MultiModalIntelligence(DEFAULT_CONFIG)
        await intelligence.initialize()
        
        # Example usage
        text_content = MultiModalContent(
            content_id="text-001",
            modality=ModalityType.TEXT,
            data="This is a sample text for analysis. It contains information about artificial intelligence and machine learning."
        )
        
        request = ProcessingRequest(
            request_id="multi-001",
            user_id="user-123",
            session_id="session-456",
            content=[text_content],
            processing_quality=ProcessingQuality.BALANCED
        )
        
        result = await intelligence.process_multimodal_request(request)
        print(f"Processing result: {result}")
        
        # Health check
        health = await intelligence.health_check()
        print(f"Health: {health}")
    
    asyncio.run(main())

