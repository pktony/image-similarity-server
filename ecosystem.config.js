module.exports = {
  apps: [{
    name: "image-similarity-server",
    script: "uv",
    args: "run python -m app.main",
    interpreter: "none",
    env: {
      PYTHONUNBUFFERED: "1"
    },
    env_production: {
      NODE_ENV: "production"
    }
  }]
};
