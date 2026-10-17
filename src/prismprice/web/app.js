import "./workbench.js";

const root = document.documentElement;
const themeButton = document.querySelector("#theme-toggle");
const storedTheme = window.localStorage.getItem("prismprice-theme");
const initialTheme = storedTheme || (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
root.dataset.theme = initialTheme;
themeButton.setAttribute("aria-pressed", String(initialTheme === "dark"));

themeButton.addEventListener("click", () => {
  const nextTheme = root.dataset.theme === "dark" ? "light" : "dark";
  root.dataset.theme = nextTheme;
  themeButton.setAttribute("aria-pressed", String(nextTheme === "dark"));
  window.localStorage.setItem("prismprice-theme", nextTheme);
});

document.querySelectorAll('a[href^="#"]').forEach((link) => {
  link.addEventListener("click", (event) => {
    const destination = document.querySelector(link.getAttribute("href"));
    if (!destination) return;
    event.preventDefault();
    destination.scrollIntoView({ behavior: "smooth", block: "start" });
    history.replaceState(null, "", link.getAttribute("href"));
  });
});
