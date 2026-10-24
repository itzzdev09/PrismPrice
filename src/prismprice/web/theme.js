const sunPath = "M12 3v2m0 14v2M3 12h2m14 0h2M5.64 5.64l1.42 1.42m9.88 9.88 1.42 1.42m0-12.72-1.42 1.42m-9.88 9.88-1.42 1.42M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z";
const moonPath = "M20.5 15.5A8.5 8.5 0 0 1 8.5 3.5 8.5 8.5 0 1 0 20.5 15.5Z";

function readPreference(view) {
  try {
    const preference = view.localStorage.getItem("prismprice-theme");
    return preference === "dark" || preference === "light" ? preference : null;
  } catch {
    return null;
  }
}

export function initTheme(documentRoot = document) {
  const view = documentRoot.defaultView || window;
  const root = documentRoot.documentElement;
  const button = documentRoot.querySelector("#theme-toggle");
  const icon = button.querySelector("path");
  const themeColor = documentRoot.querySelector('meta[name="theme-color"]');
  const preference = view.matchMedia("(prefers-color-scheme: dark)");
  let manuallySelected = readPreference(view) !== null;

  function apply(theme, persist = false) {
    root.dataset.theme = theme;
    button.setAttribute("aria-pressed", String(theme === "dark"));
    button.setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} appearance`);
    icon.setAttribute("d", theme === "dark" ? moonPath : sunPath);
    themeColor.content = theme === "dark" ? "#17171b" : "#f5f5f7";
    if (persist) {
      manuallySelected = true;
      try {
        view.localStorage.setItem("prismprice-theme", theme);
      } catch {
        // Keep the selected theme for this visit when storage is unavailable.
      }
    }
  }

  apply(readPreference(view) || (preference.matches ? "dark" : "light"));
  button.addEventListener("click", () => {
    apply(root.dataset.theme === "dark" ? "light" : "dark", true);
  });
  preference.addEventListener("change", (event) => {
    if (!manuallySelected) apply(event.matches ? "dark" : "light");
  });
}
