# KrishiMitra

An agricultural assistance application with React Native frontend and FastAPI backend.

## Setup Instructions

### Backend Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Create a `.env` file in the root directory
   - Add your Google Cloud credentials:
     ```
     GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
     ```

5. Run the backend server:
   ```bash
   uvicorn main:app --reload
   ```

### Deployment

This project is configured for deployment on Railway. Follow these steps:

1. Push your code to GitHub (excluding sensitive files)
2. Create a new project on Railway
3. Connect your GitHub repository
4. Set up the following environment variables in Railway:
   - `GOOGLE_APPLICATION_CREDENTIALS` (as a string)
   - Any other required environment variables

### Important Notes

- Never commit sensitive files (credentials, .env) to the repository
- Large data files (chroma_data, stores) should be handled separately
- For local development, place the required files in the appropriate directories 
