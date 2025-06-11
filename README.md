# scientific-research-monk
This project is to create a front-end screening web page and a large model based on a knowledge base, aiming to provide a dual-selection platform for college students and teachers in the scientific research field. Currently, the complete front-end code and large model code are available.

中文指导：

首先请通过这个链接进行下载：https://nodejs.org/zh-cn/download

结合网上安装教程进行安装完成之后，在本地打开cmd

前提： Node.js 和 npm（或 yarn）已经成功安装在你的系统上。

# 教师筛选系统项目搭建指南（基于 Node.js + Vite + React + Tailwind + Shadcn UI）

> 本指南详细说明了从安装 Node.js 后，逐步搭建教师筛选系统 Web 项目的完整流程，适用于初学者或团队文档记录。

---

## 📦 1. 安装 Node.js 与初始化项目

### ✅ 安装 Node.js

前往 [https://nodejs.org/](https://nodejs.org/) 下载并安装 LTS 版本。安装完成后在终端验证：

```bash
node -v
npm -v
```

### ✅ 使用 Vite 创建 React 项目

```bash
npm create vite@latest teacher-query --template react
cd teacher-query
npm install
```

---

## 🧱 2. 安装 Tailwind CSS

### ✅ 安装 Tailwind 及其依赖

```bash
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

### ✅ 配置 Tailwind (`tailwind.config.js`)

```js
content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
```

### ✅ 初始化 CSS 文件 (`src/index.css`)

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

---

## 🎨 3. 安装并初始化 Shadcn UI

### ✅ 初始化 Shadcn UI

```bash
npx shadcn-ui@latest init
```

按提示完成以下选项：

- 使用 TypeScript：否（如果你是用 `.jsx`）
- 样式：Default 或 New York
- Base color：slate
- Global CSS 文件路径：`src/index.css`
- 组件路径别名：`@/components`
- 工具函数别名：`@/lib/utils`
- 使用 Server Components：否

---

## 🧩 4. 添加所需 Shadcn UI 组件

```bash
npx shadcn-ui@latest add card
npx shadcn-ui@latest add input
npx shadcn-ui@latest add button
npx shadcn-ui@latest add select
npx shadcn-ui@latest add table
npx shadcn-ui@latest add label
```

如需分页功能：

```bash
npx shadcn-ui@latest add pagination
```

---

## 📁 5. 项目结构建议

```plaintext
teacher-query/
├── src/
│   ├── components/
│   │   └── ui/             # Shadcn UI 组件目录
│   ├── pages/
│   │   └── TeacherQueryPage.jsx
│   ├── App.jsx
│   ├── main.jsx
│   └── index.css
├── tailwind.config.js
├── package.json
```

---

## ⚙️ 6. 编写教师筛选功能页面（TeacherQueryPage.jsx）

### ✅ 包含以下功能模块：

- 使用 `useState` 定义筛选条件和原始教师数据
- 使用 `useEffect` 监听筛选条件变化并执行 `.filter()` 逻辑
- 渲染页面筛选栏和表格数据
- 使用 Shadcn 的 `Card`, `Select`, `Input`, `Button`, `Table` 等组件
- 提供“重置筛选”按钮

### ✅ 示例字段：

```js
const [allTeachers] = useState([
  {
    id: 1,
    university: "复旦大学",
    researchDirection: "嵌入式系统",
    department: "计算机学院",
    teacher: "陈伟男",
    tag: "论文多",
    email: "wnchen@fudan.edu.cn",
    status: "已上线"
  },
  ...
]);
```

---

## 🔗 7. 修改 App.jsx 入口文件

```jsx
import TeacherQueryPage from './pages/TeacherQueryPage';
import './index.css';

function App() {
  return <TeacherQueryPage />;
}

export default App;
```

---

## 🚀 8. 启动项目

```bash
npm run dev
```

浏览器访问： [http://localhost:5173](http://localhost:5173)

---

## 🧪 9. 常见问题排查

| 问题 | 解决方式 |
|------|----------|
| 页面白屏 | 检查控制台错误、重启 Vite、检查筛选组件配置 |
| 筛选无效 | 检查 `useEffect` 依赖项是否完整 |
| 样式无效 | 检查 Tailwind content 配置、CSS 引入是否正确 |
| Select 报错 | 确保无 `value=""` 的 SelectItem，避免空字符串 |
