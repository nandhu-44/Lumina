# Lumina - AI Art Generator

> 🎨 A modern AI art generator with real-time prompt enhancement and generation progress tracking. Built with Stable Diffusion, FastAPI, and Next.js. Features GPT-4/Gemini Pro prompt enhancement, WebSocket updates, and content safety validation.

Lumina is an AI-powered image generation system that uses Stable Diffusion with prompt enhancement capabilities.

## Features

- Advanced prompt enhancement using GPT-4 OR Gemini Pro
- Real-time generation progress updates via WebSocket
- Content safety validation for both prompts and generated images
- Customizable generation parameters
- Modern responsive UI with animations

## Prerequisites

- Python 3.10+ _(Python 3.12.0)_
- Node.js 18+ _(v22.12.0)_
- CUDA-compatible GPU (recommended)
- API Keys:
  - OpenAI API key
  - Google Gemini API key
  - Hugging Face token

## Backend Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/nandhu-44/Lumina.git
   cd Lumina
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # Linux/MacOS
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install PyTorch with CUDA support (if you have a compatible GPU) from the [pytorch official website](https://pytorch.org/get-started/locally/)

   ```bash
   pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

5. Create a `.env` file in the root directory:

   ```env
   OPENAI_API_KEY=your_openai_key
   GEMINI_API_KEY=your_gemini_key
   HF_TOKEN=your_huggingface_token
   ```

6. Start the backend server:

   ```bash
   uvicorn main:app
   ```

The server will start at `http://localhost:8000`

## Frontend Setup

For frontend setup instructions, please refer to the [web/README.md](web/README.md) file.

---
