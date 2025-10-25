# AI Dream Generator with Gesture Control

An innovative system that combines computer vision, gesture recognition, and AI text generation to create unique dreams based on hand gestures. Make two different animal shadow puppets, and watch as AI generates a new dream narrative that combines both creatures.

## 🌟 Features

- **Gesture Recognition**: Detects 10 different animal hand gestures using ML5.js
- **AI Dream Generation**: Uses Hugging Face transformers to generate coherent dream narratives
- **Real-time Processing**: Combines two gestures to create unique dream combinations
- **Beautiful Interface**: Immersive web interface with visual feedback
- **Fallback System**: Works even without trained models using existing dream data

## 🎭 Supported Animal Gestures

The system can detect these animal hand gestures:
- 🐱 **Cat** - Claw gesture
- 🐰 **Rabbit** - Bunny ears
- 🐴 **Horse** - Horse head silhouette
- 🐍 **Snake** - Slithering motion
- 🐘 **Elephant** - Trunk gesture
- 🦌 **Deer** - Antlers
- 🐟 **Fish** - Swimming motion
- 🌸 **Flower** - Blooming gesture
- 🐕 **Dog** - Paw gesture
- 🐦 **Bird** - Flying motion

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

```bash
# Clone or download the project files
cd data-archive-project

# Run the automated setup script
python setup_and_run.py --all
```

This will:
1. Install all dependencies
2. Train the AI model
3. Start the web server

### Option 2: Manual Setup

1. **Install Dependencies**
   ```bash
   pip install -e .
   ```

2. **Train the AI Model**
   ```bash
   python dream_generator.py --train --epochs 3 --batch-size 4
   ```

3. **Start the Server**
   ```bash
   python dream_api_server.py
   ```

4. **Open Your Browser**
   Navigate to `http://localhost:5000`

## 🎮 How to Use

1. **Open the Interface**: Go to `http://localhost:5000` in your browser
2. **Allow Camera Access**: The system needs camera access for gesture detection
3. **Make Gestures**: Hold up your hand and make two different animal gestures
4. **Wait for Generation**: The AI will generate a new dream combining both animals
5. **Enjoy Your Dream**: Read the generated dream narrative

### Keyboard Shortcuts
- **R**: Reset gesture combination
- **G**: Generate a random dream (for testing)

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Camera Input  │───▶│  ML5.js Gesture  │───▶│   Flask API     │
│                 │    │   Detection      │    │   Server        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌──────────────────┐            │
│   Dream Output  │◀───│  Hugging Face    │◀───────────┘
│   Display       │    │  Text Generator  │
└─────────────────┘    └──────────────────┘
```

### Components

1. **Frontend** (`index_enhanced.html`, `sketch_enhanced.js`)
   - Web interface with camera integration
   - Real-time gesture detection using ML5.js
   - Visual feedback and status indicators

2. **Backend API** (`dream_api_server.py`)
   - Flask server handling dream generation requests
   - RESTful API endpoints
   - Error handling and fallback systems

3. **AI Model** (`dream_generator.py`)
   - Hugging Face transformer-based text generation
   - Custom training on dream data
   - Fallback generation using existing data

4. **Data Processing**
   - JSON dream data from Reddit posts
   - Preprocessing and combination logic
   - Training data synthesis

## 📊 API Endpoints

- `GET /` - Main web interface
- `POST /generate_dream` - Generate dream for specific animals
- `GET /generate_random_dream` - Generate random dream
- `GET /health` - Health check
- `GET /animals` - List available animals
- `GET /stats` - System statistics

### Example API Usage

```bash
# Generate a specific dream
curl -X POST http://localhost:5000/generate_dream \
  -H "Content-Type: application/json" \
  -d '{"animal1": "cat", "animal2": "bird"}'

# Generate a random dream
curl http://localhost:5000/generate_random_dream
```

## 🔧 Configuration

### Model Training Parameters

```python
# In dream_generator.py
training_args = {
    'epochs': 3,           # Number of training epochs
    'batch_size': 4,       # Training batch size
    'learning_rate': 5e-5, # Learning rate
    'model_name': 'microsoft/DialoGPT-medium'  # Base model
}
```

### Gesture Detection Settings

```javascript
// In sketch_enhanced.js
const detectionCooldown = 2000;  // 2 seconds between detections
const maxCombinationLength = 2;  // Number of gestures to combine
```

## 🛠️ Development

### File Structure

```
data-archive-project/
├── dream_generator.py          # AI model training and generation
├── dream_api_server.py         # Flask API server
├── setup_and_run.py           # Automated setup script
├── sketch_enhanced.js         # Enhanced gesture detection
├── index_enhanced.html        # Enhanced web interface
├── dreams_*_posts.json        # Dream data files
├── images/                    # Dream images
├── other/                     # Animal reference images
└── pyproject.toml            # Dependencies
```

### Adding New Gestures

1. **Add Detection Logic** in `sketch_enhanced.js`:
   ```javascript
   if (/* your gesture condition */) {
       detectedAnimal = "your_animal";
   }
   ```

2. **Update Animal List** in both JavaScript and Python files:
   ```javascript
   const animals = ["deer", "cat", "dog", /* add your animal */];
   ```

3. **Add Dream Data** - Create `dreams_your_animal_posts.json`

### Customizing Dream Generation

Modify the `_create_combined_dream` method in `DreamDataProcessor` to change how dreams are combined:

```python
def _create_combined_dream(self, dream1, dream2, animal1, animal2):
    # Your custom combination logic here
    pass
```

## 🐛 Troubleshooting

### Common Issues

1. **Camera Not Working**
   - Ensure HTTPS or localhost
   - Check browser permissions
   - Try refreshing the page

2. **Model Training Fails**
   - Check available memory (recommend 8GB+)
   - Reduce batch size in training arguments
   - Use smaller model like `microsoft/DialoGPT-small`

3. **API Server Won't Start**
   - Check if port 5000 is available
   - Install missing dependencies: `pip install -e .`
   - Check Python version (3.8+ required)

4. **Gesture Detection Not Working**
   - Ensure good lighting
   - Keep hand centered in camera view
   - Check browser console for errors

### Performance Optimization

- **For Training**: Use GPU if available, reduce batch size for CPU
- **For Inference**: The fallback system is faster than trained models
- **For Gesture Detection**: Reduce video resolution for better performance

## 📈 Future Enhancements

- [ ] Voice narration of generated dreams
- [ ] Dream image generation using DALL-E/Stable Diffusion
- [ ] Dream sharing and social features
- [ ] Mobile app version
- [ ] More gesture types and animals
- [ ] Dream interpretation and analysis
- [ ] User dream history and favorites

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Please check the license file for details.

## 🙏 Acknowledgments

- **ML5.js** for gesture detection capabilities
- **Hugging Face** for transformer models and tools
- **Reddit r/Dreams community** for the dream data
- **P5.js** for creative coding framework

## 📞 Support

If you encounter issues or have questions:
1. Check the troubleshooting section
2. Review the browser console for errors
3. Check the server logs for backend issues
4. Open an issue with detailed error information

---

**Happy Dreaming!** 🌙✨



