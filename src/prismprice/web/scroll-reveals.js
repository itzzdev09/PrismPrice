export function initScrollReveals(documentRoot = document) {
  const view = documentRoot.defaultView || window;
  const targets = [...documentRoot.querySelectorAll("[data-reveal]")];
  if (!targets.length || !view.IntersectionObserver || view.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

  documentRoot.documentElement.classList.add("reveal-ready");
  const observer = new view.IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("is-visible");
      observer.unobserve(entry.target);
    });
  }, { threshold: 0.12, rootMargin: "0px 0px -36px 0px" });

  targets.forEach((target) => observer.observe(target));
}
