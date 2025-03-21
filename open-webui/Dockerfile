FROM ghcr.io/open-webui/open-webui:main

# Expose the port that the application uses
EXPOSE 8080

# Set up a volume for persistent data
VOLUME ["/app/backend/data"]

# Keep the container running
CMD ["sh", "-c", "cd /app && node server.js"]