/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        "bg-base":   "var(--bg-base)",
        "bg-panel":  "var(--bg-panel)",
        "bg-elev":   "var(--bg-elev)",
        "bg-hover":  "var(--bg-hover)",
        "bg-active": "var(--bg-active)",

        "border-subtle": "var(--border-subtle)",
        "border-strong": "var(--border-strong)",

        fg:           "var(--fg)",
        "fg-secondary": "var(--fg-secondary)",
        "fg-tertiary":  "var(--fg-tertiary)",
        "fg-muted":     "var(--fg-muted)",

        accent:        "var(--accent)",
        "accent-soft": "var(--accent-soft)",

        "sig-red":     "var(--sig-red)",
        "sig-amber":   "var(--sig-amber)",
        "sig-green":   "var(--sig-green)",
        "sig-blue":    "var(--sig-blue)",
        "sig-emerald": "var(--sig-emerald)",
        "sig-yellow":  "var(--sig-yellow)",
        "sig-violet":  "var(--sig-violet)",
      },
      fontFamily: {
        sans: "var(--font-sans)",
        mono: "var(--font-mono)",
      },
      borderRadius: {
        DEFAULT: "var(--radius)",
        lg: "var(--radius-lg)",
      },
    },
  },
  plugins: [],
};
