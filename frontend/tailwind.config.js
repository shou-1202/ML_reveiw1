/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#14161A",
        surface: "#1B1E23",
        raised: "#22262C",
        border: "#2B2F36",
        ink: "#E7E9EC",
        muted: "#8B9099",
        faint: "#5C6169",
        amber: {
          DEFAULT: "#E7A94C",
          dim: "#8A6B37",
        },
        teal: {
          DEFAULT: "#4FB4A8",
          dim: "#33544F",
        },
        danger: "#E0645C",
      },
      fontFamily: {
        sans: ["'IBM Plex Sans'", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "monospace"],
      },
      borderRadius: {
        sm: "3px",
        DEFAULT: "4px",
        md: "6px",
      },
      boxShadow: {
        panel: "0 1px 0 0 rgba(0,0,0,0.4)",
      },
    },
  },
  plugins: [],
};
