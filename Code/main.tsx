import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App.jsx'; // 确认路径正确
import './index.css'; // 确保全局样式在这里也被导入

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
