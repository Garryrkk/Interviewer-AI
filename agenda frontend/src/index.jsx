import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css"; // only if you have styles

// Make sure your public/index.html has <div id="root"></div>
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
