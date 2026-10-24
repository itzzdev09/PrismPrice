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
  const themeImages = [...documentRoot.querySelectorAll("[data-theme-image-light][data-theme-image-dark]")];
  const themeColor = documentRoot.querySelector('meta[name="theme-color"]');
  const preference = view.matchMedia("(prefers-color-scheme: dark)");
  let manuallySelected = readPreference(view) !== null;

  function apply(theme, persist = false) {
    root.dataset.theme = theme;
    button.setAttribute("aria-pressed", String(theme === "dark"));
    button.setAttribute("aria-label", `Switch to ${theme === "dark" ? "light" : "dark"} mode`);
    themeColor.content = theme === "dark" ? "#17171b" : "#f5f5f7";
    themeImages.forEach((image) => {
      image.src = image.dataset[theme === "dark" ? "themeImageDark" : "themeImageLight"];
    });
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
