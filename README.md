# LIAT - Live Intelligence Analysis Tool

**Multi-View Player Tracking System for Sports Videos**

This system tracks players across two different camera angles (broadcast and tacticam) and assigns consistent player IDs using YOLOv11 detection and advanced multi-feature matching.

## 🎯 Overview

The LIAT system solves the challenging problem of maintaining consistent player identities across multiple camera views in sports videos. It combines computer vision, machine learning, and geometric alignment techniques to achieve robust player tracking.

### Key Features

- **Multi-View Detection**: Uses YOLOv11 to detect players in both broadcast and tacticam videos
- **Rich Feature Extraction**: Extracts appearance, position, size, and texture features
- **Intelligent Matching**: Matches players across views using multi-feature similarity scoring
- **Temporal Consistency**: Ensures player IDs remain stable across frames
- **Robust Alignment**: Handles different camera angles and perspectives
- **Comprehensive Output**: Exports results in CSV and JSON formats

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Videos: `broadcast.mp4` and `tacticam.mp4`
- YOLOv11 model: `best.pt`

### Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the complete pipeline:
```bash
python run_pipeline.py
```

### Basic Usage

```bash
# Use default files (broadcast.mp4, tacticam.mp4, best.pt)
python run_pipeline.py

# Use custom files
python run_pipeline.py --broadcast my_broadcast.mp4 --tacticam my_tacticam.mp4

# Skip temporal tracking for faster processing
python run_pipeline.py --skip-temporal

# Specify output directory
python run_pipeline.py --output my_results/
```

## 🔧 System Architecture

### Pipeline Components

1. **Player Detection (`player_tracker.py`)**
   - Detects players using YOLOv11
   - Extracts rich features (color, texture, position, size)
   - Performs cross-view matching with geometric alignment

2. **Temporal Tracking (`temporal_tracker.py`)**
   - Maintains player identity across frames
   - Uses motion consistency and appearance similarity
   - Handles occlusions and temporary disappearances

3. **Pipeline Runner (`run_pipeline.py`)**
   - Orchestrates the complete workflow
   - Handles error checking and logging
   - Generates comprehensive reports

### Feature Extraction

- **Appearance Features**:
  - HSV color histograms
  - Texture features (gradient magnitude)
  - Player size and aspect ratio

- **Position Features**:
  - Relative position within frame
  - Geometric alignment between views
  - Motion consistency over time

- **Matching Algorithm**:
  - Multi-feature similarity scoring
  - Hungarian algorithm for optimal assignment
  - Temporal consistency constraints

## 📁 File Structure

```
LIAT/
├── run_pipeline.py          # Main pipeline runner
├── player_tracker.py        # Core player tracking system
├── temporal_tracker.py      # Temporal consistency tracker
├── requirements.txt         # Python dependencies
├── README.md               # This documentation
├── best.pt                 # YOLOv11 model file
├── broadcast.mp4           # Broadcast camera video
├── tacticam.mp4           # Tactical camera video
└── results/               # Output directory
    ├── player_tracking_results.csv
    ├── player_tracking_results.json
    ├── temporal_tracking_results.csv
    ├── temporal_tracking_results.json
    └── pipeline_summary.json
```

## 📊 Output Format

### CSV Output

The system generates CSV files with the following columns:

- `frame`: Frame number
- `view`: Camera view (broadcast/tacticam)
- `player_id`: Consistent player identifier
- `bbox`: Bounding box coordinates [x1, y1, x2, y2]
- `confidence`: Detection confidence score
- `position`: Relative position [x, y] in frame
- `match_similarity`: Similarity score for cross-view matches

### JSON Output

Detailed JSON files include:
- Complete detection information
- Feature vectors for each detection
- Matching scores and metadata
- Pipeline configuration and statistics

## 🎛️ Advanced Configuration

### Tracking Parameters

- **Similarity Threshold**: Minimum similarity for matching (default: 0.3)
- **Max Missing Frames**: Frames before track is lost (default: 5)
- **Confidence Threshold**: Minimum detection confidence (default: 0.3)

### Feature Weights

- Position similarity: 40%
- Appearance similarity: 30%
- Size similarity: 20%
- Motion consistency: 10%

## 🔍 Algorithm Details

### Cross-View Matching

1. **Geometric Alignment**: Uses Procrustes analysis to align player positions between views
2. **Feature Similarity**: Computes multi-dimensional similarity scores
3. **Optimal Assignment**: Uses Hungarian algorithm for one-to-one matching
4. **Temporal Consistency**: Maintains tracks across frames using motion models

### Handling Edge Cases

- **Occlusions**: Temporary track loss with re-identification
- **Different Player Counts**: Handles varying numbers of visible players
- **Camera Movement**: Robust to small camera adjustments
- **Lighting Changes**: HSV color space provides illumination invariance

## 📈 Performance Optimization

### Speed Optimization

- Frame skipping for real-time processing
- Efficient feature extraction using OpenCV
- Optimized similarity computation
- Parallel processing capabilities

### Memory Management

- Streaming video processing
- Efficient data structures
- Garbage collection optimization
- Configurable history length

## 🛠️ Troubleshooting

### Common Issues

1. **Missing Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **CUDA/GPU Issues**:
   - Install appropriate PyTorch version
   - Check GPU compatibility

3. **Video Format Issues**:
   - Ensure videos are in MP4 format
   - Check video codec compatibility

4. **Model Loading Issues**:
   - Verify YOLOv11 model file exists
   - Check model format compatibility

### Debug Mode

```bash
# Run with detailed logging
python run_pipeline.py --verbose

# Skip dependency checks for debugging
python run_pipeline.py --skip-checks
```

## 🤝 Contributing

We welcome contributions! Please see our guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the troubleshooting section
- Review the documentation

## 🔮 Future Enhancements

- Real-time processing capabilities
- Support for more than two camera views
- Advanced motion prediction models
- Integration with sports analytics platforms
- Web-based visualization dashboard

---

**LIAT** - Making sports video analysis intelligent and accessible.
