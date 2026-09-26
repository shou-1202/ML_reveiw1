import { Routes, Route } from "react-router-dom";
import Navbar from "./components/Navbar";
import Home from "./pages/Home";
import Preprocessing from "./pages/Preprocessing";
import RegressionLab from "./pages/RegressionLab";

export default function App() {
  return (
    <div className="min-h-screen bg-bg">
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/preprocessing" element={<Preprocessing />} />
        <Route path="/regression" element={<RegressionLab />} />
      </Routes>
    </div>
  );
}
