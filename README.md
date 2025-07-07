# Grading System API

A Flask-based API that grades student answers against model answers using BLEU and ROUGE scores, enhanced with AI-powered feedback from Google Gemini.

## Features

- **BLEU Score**: Measures precision of student answers
- **ROUGE Score**: Measures recall and content coverage
- **AI Feedback**: Generates constructive feedback using Google Gemini
- **RESTful API**: Easy to integrate with any frontend
- **CORS Support**: Can be used with web applications

## Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd Grades_Scoring
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. **Get Google API Key**:
   - Go to [Google AI Studio](https://aistudio.google.com/)
   - Create a new API key
   - Copy the key to your `.env` file

## Usage

### Running the API

```bash
python API.py
```

The API will start on `http://localhost:5000`

### API Endpoints

#### GET /
- **Description**: Home endpoint with API information
- **Response**: JSON with API details

#### GET /health
- **Description**: Health check endpoint
- **Response**: `{"status": "healthy", "message": "API is running"}`

#### POST /grade
- **Description**: Grade student answer against model answer
- **Request Body**:
  ```json
  {
    "student_answer": "Your student's answer here",
    "model_answer": "The correct/expected answer here"
  }
  ```
- **Response**:
  ```json
  {
    "student_answer": "Student's answer",
    "model_answer": "Model answer",
    "scores": {
      "bleu": 0.1234,
      "rouge-1": 0.5678,
      "rouge-2": 0.3456,
      "rouge-l": 0.4567
    },
    "feedback": "AI-generated feedback...",
    "timestamp": "2025-07-08T10:30:00"
  }
  ```

## Testing with Postman

### 1. Health Check
- **Method**: GET
- **URL**: `http://localhost:5000/health`
- **Expected Response**: Status 200 with health message

### 2. Grade Answer
- **Method**: POST
- **URL**: `http://localhost:5000/grade`
- **Headers**: 
  - `Content-Type: application/json`
- **Body** (raw JSON):
  ```json
  {
    "student_answer": "Photosynthesis is when plants make food using sunlight",
    "model_answer": "Photosynthesis is the process by which plants convert light energy, carbon dioxide, and water into glucose and oxygen using chlorophyll"
  }
  ```

### 3. Postman Collection
You can import this collection into Postman:

```json
{
  "info": {
    "name": "Grading System API",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "item": [
    {
      "name": "Health Check",
      "request": {
        "method": "GET",
        "header": [],
        "url": {
          "raw": "http://localhost:5000/health",
          "protocol": "http",
          "host": ["localhost"],
          "port": "5000",
          "path": ["health"]
        }
      }
    },
    {
      "name": "Grade Answer",
      "request": {
        "method": "POST",
        "header": [
          {
            "key": "Content-Type",
            "value": "application/json"
          }
        ],
        "body": {
          "mode": "raw",
          "raw": "{\n  \"student_answer\": \"Photosynthesis is when plants make food using sunlight\",\n  \"model_answer\": \"Photosynthesis is the process by which plants convert light energy, carbon dioxide, and water into glucose and oxygen using chlorophyll\"\n}"
        },
        "url": {
          "raw": "http://localhost:5000/grade",
          "protocol": "http",
          "host": ["localhost"],
          "port": "5000",
          "path": ["grade"]
        }
      }
    }
  ]
}
```

## Score Interpretation

### BLEU Score (0.0 - 1.0)
- **0.0-0.3**: Poor similarity
- **0.3-0.5**: Fair similarity
- **0.5-0.7**: Good similarity
- **0.7-1.0**: Excellent similarity

### ROUGE Score (0.0 - 1.0)
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence
- Higher scores indicate better content coverage

## Error Handling

The API handles various error cases:
- Missing JSON data
- Missing required fields
- Invalid input
- Internal server errors
- Google API failures

## Development

### Project Structure
```
Grades_Scoring/
├── API.py              # Main Flask application
├── notebook.ipynb      # Jupyter notebook for testing
├── tools.py           # Additional utility functions
├── requirements.txt   # Python dependencies
├── README.md         # This file
└── .env              # Environment variables (create this)
```

### Adding New Features
1. Add new endpoints in `API.py`
2. Update the README with new endpoint documentation
3. Test with Postman
4. Update requirements.txt if new packages are needed

## Troubleshooting

### Common Issues

1. **Missing Google API Key**:
   - Ensure `.env` file exists with valid `GOOGLE_API_KEY`
   - Check Google AI Studio for API key status

2. **NLTK Data Missing**:
   - The API automatically downloads required NLTK data
   - If issues persist, manually run: `nltk.download('punkt')`

3. **Port Already in Use**:
   - Change the port in `API.py`: `app.run(port=5001)`
   - Or kill the process using port 5000

4. **CORS Issues**:
   - The API includes CORS support
   - Ensure `flask-cors` is installed

## License

This project is for educational purposes.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request
