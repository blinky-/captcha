Dependencies:

``pip install opencv-python numpy requests``

Usage:
```python
import captcha_reader

# Recognize from file
digits = captcha_reader.recognize_from_file(file)

# Or from URL
digits = captcha_reader.recognize_from_url(url)
```

To run tests execute `main.py`