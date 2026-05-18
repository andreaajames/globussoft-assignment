# Globussoft Data Science Assignment

A two-part Python project covering web scraping and computer vision.

---

## Project Structure

```
├── task1_amazon_scraper.ipynb     # Amazon laptop scraper
├── task2_training.ipynb           # Face authentication - model setup
├── task2_testing.ipynb            # Face authentication - inference
├── app.py                         # FastAPI service for face verification
├── requirements.txt               # All dependencies
├── model_config.json              # Generated after running training notebook
├── sample_images/
│   ├── person1_a.jpg
│   └── person1_b.jpg
└── README.md
```

---

## Task 1 — Amazon Laptop Scraper

Scrapes laptop listings from amazon.in and saves the results to a timestamped CSV file.

### Fields Collected

| Field | Description |
|-------|-------------|
| Title | Product name |
| Image URL | Link to product image |
| Rating | Star rating out of 5 |
| Price | Listed price in INR |
| Result Type | Whether the listing is an Ad or Organic result |

### How to Run

```bash
jupyter notebook task1_amazon_scraper.ipynb
```

Run all cells. The output CSV will be saved automatically in the same folder with a filename like:

```
amazon_laptops_20250518_143022.csv
```

---

## Task 2 — Face Authentication (Face Verification)

A face verification system that accepts two images and determines whether they belong to the same person. Built using **FaceNet** embeddings via the **DeepFace** library.

### How It Works

1. Both images are passed through the FaceNet model to extract 128-dimension face embeddings
2. Cosine similarity is computed between the two embeddings
3. If the similarity score is above the threshold, the result is `same person`, otherwise `different person`
4. Bounding box coordinates of detected faces are also returned

### Output Format

```json
{
  "verification_result": "same person",
  "similarity_score": 0.8741,
  "bounding_boxes": {
    "image_1": [{"x": 100, "y": 80, "w": 120, "h": 130}],
    "image_2": [{"x": 90,  "y": 75, "w": 115, "h": 125}]
  }
}
```

### Step 1 — Run Training Notebook

```bash
jupyter notebook task2_training.ipynb
```

This loads the FaceNet model and saves `model_config.json`. Only needs to be done once.

### Step 2 — Run Testing Notebook

Place two face images inside the `sample_images/` folder, then:

```bash
jupyter notebook task2_testing.ipynb
```

Update the image paths in the last cell if your filenames are different.

### Step 3 — Run FastAPI Service

```bash
uvicorn app:app --reload
```

The API will be live at `http://127.0.0.1:8000`

Open `http://127.0.0.1:8000/docs` in your browser for the interactive Swagger UI where you can upload images and test the endpoint directly.

#### API Endpoint

**POST** `/verify`

| Parameter | Type | Description |
|-----------|------|-------------|
| image1 | file | First face image |
| image2 | file | Second face image |

---

## Installation

```bash
pip install -r requirements.txt
```

> Note: First-time setup downloads the FaceNet model weights (~90MB). This happens automatically.

---

## Tech Stack

- Python 3.10
- DeepFace + FaceNet
- OpenCV
- FastAPI
- BeautifulSoup4
- Pandas
