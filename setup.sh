#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Manhwa Generator environment...${NC}"

# Check if .env exists
if [ -f .env ]; then
    echo -e "${YELLOW}Warning: .env file already exists. Creating backup...${NC}"
    cp .env .env.backup
fi

# Copy example file
cp .env.example .env

echo -e "${GREEN}Created .env file from template${NC}"
echo -e "${YELLOW}Please edit .env file and add your API tokens and preferences${NC}"

# Create output directory
mkdir -p output

# Check Python environment
if [ ! -d "menv" ]; then
    echo -e "${GREEN}Creating Python virtual environment...${NC}"
    python3 -m venv menv
    echo -e "${GREEN}Installing requirements...${NC}"
    source menv/bin/activate
    pip install -r requirements.txt
fi

# Create fonts directory and download NanumGothic if needed
mkdir -p fonts
if [ ! -f "fonts/NanumGothic.ttf" ]; then
    echo -e "${GREEN}Downloading NanumGothic font...${NC}"
    curl -o fonts/NanumGothic.ttf https://raw.githubusercontent.com/googlefonts/nanum-gothic/main/fonts/NanumGothic-Regular.ttf
fi

echo -e "${GREEN}Setup complete!${NC}"
echo -e "Next steps:"
echo -e "1. Edit the .env file with your settings"
echo -e "2. Activate the virtual environment: ${YELLOW}source menv/bin/activate${NC}"
echo -e "3. Run the generator: ${YELLOW}python test10.py${NC}"