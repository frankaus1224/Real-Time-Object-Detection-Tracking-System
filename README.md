# Real-Time Object Detection & Tracking System

This project is a high-performance real-time object detection system built with **Python**, **YOLOv8/v11**, and **OpenCV**. It features a modular, decoupled architecture optimized for scalability and runs efficiently on **Apple Silicon (M1/M2/M3)** using MPS acceleration.

## 🚀 Features
- **Modular Decoupled Architecture**: Separation of core logic, UI, and configuration for high maintainability.
- **Real-time Detection**: Recognizes 80+ object classes (people, cars, phones, etc.).
- **Hardware Acceleration**: Optimized for Mac M1 GPU (Metal Performance Shaders).
- **OOP Architecture**: Built with a clean, class-based structure for easy expansion.
- **FPS Monitoring**: Real-time inference speed display.

## 🏗️ System Architecture
The project follows a modular design to ensure scalability and ease of integration:
```text
├── main.py              # Application entry point: Orchestrates the system flow.
├── core/                # Logic Layer: Handles YOLO inference and data parsing.
├── configs/             # Configuration Layer: Manages hardware and model settings.
├── utils/               # Utility Layer: Handles UI rendering and visualization.
└── archive/             # Legacy scripts and prototypes.

## 🛠️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/frankaus1224/Real-Time-Object-Detection-Tracking-System.git](https://github.com/frankaus1224/Real-Time-Object-Detection-Tracking-System.git)
   cd Real-Time-Object-Detection-Tracking-System

2. **Create a virtual environment:**
    python3 -m venv .venv
    source .venv/bin/activate

3. **Install dependencies:**
    pip install -r requirements.txt

## 💻 Usage
    To launch the real-time detection system:
    python main.py
    Press 's': Save a screenshot of the current frame.
    Press 'q' on the video window to safely exit the application.

📈 Project Roadmap
    [x] Phase 1: Environment setup & MPS (Metal) Optimization.

    [x] Phase 2: Modular System Architecture Redesign (Decoupling).

    [x] Phase 3: Real-time Object Counting & UI Enhancements.


## 👥 Contributors
    Member: Frank Lin (frank.lin02@sjsu.edu)

    Member: Fnu Saad (fnu.saad@sjsu.edu)
