from app import create_app

app = create_app()

if __name__ == "__main__":
    # Use 5001 on macOS; port 5000 is often used by AirPlay and returns 403
    app.run(debug=True, port=5001)
