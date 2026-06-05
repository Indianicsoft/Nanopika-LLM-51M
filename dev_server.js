const express = require('express');
const app = express();
const port = 5500;

app.use((req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cross-Origin-Resource-Policy', 'same-origin');
  next();
});

app.use(express.static('.'));

app.listen(port, () => {
  console.log(`Training Server running at http://localhost:${port}`);
});
