const desktopFrames = [
  { scale: 0.8, x: 20, y: 0 },
  { scale: 0.92, x: 12, y: 0 },
  { scale: 1.22, x: -5, y: -10 },
  { scale: 1.62, x: -28, y: -26 },
  { scale: 2.12, x: -74, y: -58 },
];

const mobileFrames = [
  { scale: 0.62, x: 0, y: 0 },
  { scale: 0.72, x: 0, y: 0 },
  { scale: 0.9, x: 0, y: -8 },
  { scale: 1.12, x: 0, y: -18 },
  { scale: 1.34, x: 0, y: -28 },
];

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function interpolateFrames(frames, progress) {
  const point = progress * (frames.length - 1);
  const index = Math.min(frames.length - 2, Math.floor(point));
  const rawFraction = point - index;
  const fraction = rawFraction * rawFraction * (3 - 2 * rawFraction);
  const from = frames[index];
  const to = frames[index + 1];

  return {
    scale: from.scale + (to.scale - from.scale) * fraction,
    x: from.x + (to.x - from.x) * fraction,
    y: from.y + (to.y - from.y) * fraction,
  };
}

export function initProductStory(documentRoot = document) {
  const view = documentRoot.defaultView || window;
  const track = documentRoot.querySelector("#story-track");
  const stage = documentRoot.querySelector("#story-stage");
  const image = documentRoot.querySelector("#story-prism");
  const steps = [...documentRoot.querySelectorAll("[data-story-step]")];
  const buttons = [...documentRoot.querySelectorAll("[data-story-go]")];

  if (!track || !stage || !image || steps.length === 0 || buttons.length !== steps.length) return;

  const reducedMotion = view.matchMedia("(prefers-reduced-motion: reduce)");
  let frameRequest = 0;

  function setActiveStep(index) {
    steps.forEach((step, stepIndex) => {
      const active = stepIndex === index;
      step.hidden = !active;
      if (active) step.removeAttribute("aria-hidden");
      else step.setAttribute("aria-hidden", "true");
    });
    buttons.forEach((button, buttonIndex) => {
      if (buttonIndex === index) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });
    stage.dataset.activeStep = String(index);
  }

  function update() {
    frameRequest = 0;
    const stickyTop = Number.parseFloat(view.getComputedStyle(stage).top) || 0;
    const travel = Math.max(1, track.offsetHeight - stage.offsetHeight);
    const distance = stickyTop - track.getBoundingClientRect().top;
    const progress = clamp(distance / travel, 0, 1);
    const index = Math.min(steps.length - 1, Math.floor(progress * steps.length));
    setActiveStep(index);

    const frames = view.matchMedia("(max-width: 740px)").matches ? mobileFrames : desktopFrames;
    const pose = reducedMotion.matches ? frames[0] : interpolateFrames(frames, progress);
    image.style.setProperty("--story-scale", pose.scale.toFixed(3));
    image.style.setProperty("--story-x", `${pose.x.toFixed(1)}px`);
    image.style.setProperty("--story-y", `${pose.y.toFixed(1)}px`);
  }

  function scheduleUpdate() {
    if (frameRequest) return;
    frameRequest = view.requestAnimationFrame(update);
  }

  function goToStep(index) {
    const stickyTop = Number.parseFloat(view.getComputedStyle(stage).top) || 0;
    const travel = Math.max(1, track.offsetHeight - stage.offsetHeight);
    const progress = (index + 0.5) / steps.length;
    const top = track.getBoundingClientRect().top + view.scrollY - stickyTop + travel * progress;
    view.scrollTo({ top, behavior: reducedMotion.matches ? "auto" : "smooth" });
  }

  buttons.forEach((button) => {
    button.addEventListener("click", () => goToStep(Number(button.dataset.storyGo)));
  });
  view.addEventListener("scroll", scheduleUpdate, { passive: true });
  view.addEventListener("resize", scheduleUpdate, { passive: true });
  reducedMotion.addEventListener("change", scheduleUpdate);
  update();
}