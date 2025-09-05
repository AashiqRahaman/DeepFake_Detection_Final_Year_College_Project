## How to Use
---

run (from project root)

test whole folder (expects fake/ and real/ inside the folder):
```bash
python Testing/test.py --model Models/my_deepfake_detector.keras --test Assets/1000_videos/test
```

Example output:
```bash
img1.png → Predicted: fake, Prob: {'fake': 0.92, 'real': 0.08}
img2.png → Predicted: real, Prob: {'fake': 0.12, 'real': 0.88}

Overall Test Accuracy: 85.71% (12/14)
```
