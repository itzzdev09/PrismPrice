import "./workbench.js";

const root = document.documentElement;
const themeButton = document.querySelector("#theme-toggle");
const themeIcon = themeButton.querySelector("path");
const themeColor = document.querySelector('meta[name="theme-color"]');
const sunPath = themeIcon.getAttribute("d");
const moonPath = "M20.5 15.5A8.5 8.5 0 0 1 8.5 3.5 8.5 8.5 0 1 0 20.5 15.5Z";
const themePreference = window.matchMedia("(prefers-color-scheme: dark)");
const readTheme = () => {
  try {
    const theme = window.localStorage.getItem("prismprice-theme");
    return theme === "dark" || theme === "light" ? theme : null;
  } catch {
    return null;
  }
};
let manuallySelectedTheme = readTheme() !== null;

function applyTheme(theme, persist = false) {
  root.dataset.theme = theme;
  themeButton.setAttribute("aria-pressed", String(theme === "dark"));
  themeButton.setAttribute("aria-label", `${theme === "dark" ? "Dark" : "Light"} appearance`);
  themeIcon.setAttribute("d", theme === "dark" ? moonPath : sunPath);
  themeColor.content = theme === "dark" ? "#171a1f" : "#f8f9fa";
  if (persist) {
    manuallySelectedTheme = true;
    try {
      window.localStorage.setItem("prismprice-theme", theme);
    } catch {
      // The theme still works for this visit when storage is unavailable.
    }
  }
}

applyTheme(readTheme() || (themePreference.matches ? "dark" : "light"));

themeButton.addEventListener("click", () => {
  const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
  applyTheme(nextTheme, true);
});

themePreference.addEventListener("change", (event) => {
  if (!manuallySelectedTheme) applyTheme(event.matches ? "dark" : "light");
});

document.querySelectorAll('a[href^="#"]').forEach((link) => {
  link.addEventListener("click", (event) => {
    const destination = document.querySelector(link.getAttribute("href"));
    if (!destination) return;
    event.preventDefault();
    const behavior = window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
    destination.scrollIntoView({ behavior, block: "start" });
    if (window.location.hash !== link.getAttribute("href")) {
      window.history.pushState(null, "", link.getAttribute("href"));
    }
  });
});
