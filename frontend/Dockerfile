FROM node:18-alpine

# Set working directory
WORKDIR /app

# Add `/app/node_modules/.bin` to $PATH
ENV PATH /app/node_modules/.bin:$PATH

# Install app dependencies
COPY package*.json ./
RUN npm install --silent
RUN npm install react-scripts@5.0.1 -g --silent

# Add app
COPY . ./

# Start app
CMD ["npm", "start"]