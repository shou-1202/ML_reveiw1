import { NavLink } from "react-router-dom";
import { FlaskConical } from "lucide-react";

const linkBase =
  "text-sm transition-colors px-3 py-1.5 rounded";

export default function Navbar() {
  return (
    <header className="sticky top-0 z-20 border-b border-border bg-bg/90 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-6xl items-center justify-between px-6">
        <NavLink to="/" className="flex items-center gap-2 text-ink">
          <FlaskConical className="h-4 w-4 text-amber" strokeWidth={1.75} />
          <span className="font-medium">Housing Price Regression Lab</span>
        </NavLink>

        <nav className="flex items-center gap-1">
          <NavLink
            to="/"
            end
            className={({ isActive }) =>
              `${linkBase} ${isActive ? "text-ink bg-raised" : "text-muted hover:text-ink"}`
            }
          >
            Home
          </NavLink>
          <NavLink
            to="/preprocessing"
            className={({ isActive }) =>
              `${linkBase} ${isActive ? "text-ink bg-raised" : "text-muted hover:text-ink"}`
            }
          >
            Dataset Preprocessing
          </NavLink>
          <NavLink
            to="/regression"
            className={({ isActive }) =>
              `${linkBase} ${isActive ? "text-ink bg-raised" : "text-muted hover:text-ink"}`
            }
          >
            Regression Lab
          </NavLink>
        </nav>
      </div>
    </header>
  );
}
