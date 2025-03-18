# Song_Slide
Slideshow player that synchronizes image transitions with real-time audio

# Audio-Synced Slideshow Player

This project is a slideshow player that synchronizes image transitions with real-time audio analysis. It supports multiple independent players, ensuring smooth operation even when closing or opening new instances.

## Features
- **Real-time audio analysis**: Detects peaks and adjusts image transitions dynamically.
- **Automatic threshold adjustment**: Sensitivity adapts based on recent history.
- **Hysteresis control**: Prevents excessive image swapping by adjusting dynamically.
- **Multiple independent players**: Run multiple slideshows simultaneously with the same audio.
- **Manual controls**: Fine-tune detection parameters using sliders.

## Screenshot
![Output](output.png)

## Installation
Ensure you have Python installed and the required dependencies:

```bash
pip install numpy sounddevice soundcard matplotlib pillow pywin32

