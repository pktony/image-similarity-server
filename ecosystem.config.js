const dotenv = require('dotenv');
const path = require('path');

// Load .env file
dotenv.config({ path: path.join(__dirname, '.env') });

module.exports = {
  apps: [{
    name: "image-similarity-server",
    script: "uv",
    args: `run uvicorn app.main:app --host ${process.env.HOST || '0.0.0.0'} --port ${process.env.PORT || '8000'}`,
    interpreter: "none",
    env: {
      PYTHONUNBUFFERED: "1"
    },
    env_production: {
      NODE_ENV: "production"
    }
  }]
};
