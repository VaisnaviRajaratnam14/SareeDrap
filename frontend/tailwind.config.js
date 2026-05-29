/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        saree: {
          50:  "#fdf4ff",
          100: "#fae8ff",
          200: "#f3c6ff",
          300: "#e996ff",
          400: "#d957ff",
          500: "#c41fff",
          600: "#a800e6",
          700: "#8900bc",
          800: "#720099",
          900: "#5e007d",
        },
        gold: {
          400: "#fbbf24",
          500: "#f59e0b",
          600: "#d97706",
        },
      },
      fontFamily: {
        display: ["'Playfair Display'", "serif"],
        body:    ["'Inter'", "sans-serif"],
      },
      backgroundImage: {
        "hero-gradient": "linear-gradient(135deg, #5e007d 0%, #a800e6 50%, #d957ff 100%)",
        "card-gradient": "linear-gradient(145deg, #1a0a2e 0%, #2d1155 100%)",
      },
      animation: {
        "pulse-slow":   "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "float":        "float 6s ease-in-out infinite",
        "shimmer":      "shimmer 2s linear infinite",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%":      { transform: "translateY(-12px)" },
        },
        shimmer: {
          "0%":   { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition:  "200% 0" },
        },
      },
    },
  },
  plugins: [],
};
