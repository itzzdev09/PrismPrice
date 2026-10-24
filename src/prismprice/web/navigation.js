export function initNavigation(documentRoot = document) {
  const view = documentRoot.defaultView || window;
  documentRoot.querySelectorAll('a[href^="#"]').forEach((link) => {
    link.addEventListener("click", (event) => {
      const destination = documentRoot.querySelector(link.getAttribute("href"));
      if (!destination) return;
      event.preventDefault();
      const behavior = view.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
      destination.scrollIntoView({ behavior, block: "start" });
      if (view.location.hash !== link.getAttribute("href")) {
        view.history.pushState(null, "", link.getAttribute("href"));
      }
    });
  });
}
