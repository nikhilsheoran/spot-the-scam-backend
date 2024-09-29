# Spot The Scam

Spot The Scam is a comprehensive fraud detection tool that leverages AI/ML to identify potentially fraudulent websites and mobile applications.

## Project Structure

The project consists of two main components:

1. Frontend: Next.js application
   - Repository: [https://github.com/nikhilsheoran/spot-the-scam](https://github.com/nikhilsheoran/spot-the-scam)

2. Backend: Flask application
   - Repository: [https://github.com/nikhilsheoran/spot-the-scam-backend](https://github.com/nikhilsheoran/spot-the-scam-backend)

## Prerequisites

- Node.js (v14 or later)
- Python 3.8 or later
- npm or yarn
- pip

## Setting Up the Frontend (Next.js)

1. Clone the repository:
   ```
   git clone https://github.com/nikhilsheoran/spot-the-scam.git
   cd spot-the-scam
   ```

2. Install dependencies:
   ```
   npm install
   ```
   or if you're using yarn:
   ```
   yarn install
   ```

3. Set up environment variables:
   Create a `.env.local` file in the root directory and add necessary environment variables.

4. Run the development server:
   ```
   npm run dev
   ```
   or
   ```
   yarn dev
   ```

The frontend will be available at `http://localhost:3000`.

## Setting Up the Backend (Flask)

1. Clone the repository:
   ```
   git clone https://github.com/nikhilsheoran/spot-the-scam-backend.git
   cd spot-the-scam-backend
   ```

2. Create and activate a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory and add necessary environment variables.

5. Run the Flask application:
   ```
   python3 app.py
   ```

The backend API will be available at `http://localhost:5000`.

## Running the Complete Application

To run the complete Spot The Scam application:

1. Start the Flask backend server.
2. In a separate terminal, start the Next.js frontend development server.
3. Access the application through the frontend URL (`http://localhost:3000`).

Ensure both servers are running concurrently for full functionality.

## Additional Information

- The frontend uses TypeScript, Tailwind CSS, shadcn, and MagicUI for the user interface.
- The backend utilizes various machine learning models for fraud detection.
- API authentication is implemented using bearer tokens.

For more detailed information, refer to the individual README files in each repository.