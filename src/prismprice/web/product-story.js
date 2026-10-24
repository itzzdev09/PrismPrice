const FRAME_COUNT = 180;

const cameraKeyframes = [
  { scale: 0.82, x: 0.68, y: 0.5, rotate: -0.012 },
  { scale: 0.96, x: 0.67, y: 0.49, rotate: -0.005 },
  { scale: 1.2, x: 0.64, y: 0.47, rotate: 0.006 },
  { scale: 1.57, x: 0.59, y: 0.46, rotate: 0.012 },
  { scale: 2.18, x: 0.53, y: 0.49, rotate: 0.002 },
];

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function interpolateCamera(progress) {
  const point = progress * (cameraKeyframes.length - 1);
  const index = Math.min(cameraKeyframes.length - 2, Math.floor(point));
  const fraction = point - index;
  const eased = fraction * fraction * (3 - 2 * fraction);
  const start = cameraKeyframes[index];
  const end = cameraKeyframes[index + 1];

  return {
    scale: start.scale + (end.scale - start.scale) * eased,
    x: start.x + (end.x - start.x) * eased,
    y: start.y + (end.y - start.y) * eased,
    rotate: start.rotate + (end.rotate - start.rotate) * eased,
  };
}

const frameSequence = Array.from({ length: FRAME_COUNT }, (_, index) => ({
  index,
  camera: interpolateCamera(index / (FRAME_COUNT - 1)),
}));

function loadImage(view, source) {
  return new Promise((resolve, reject) => {
    const image = new view.Image();
    let settled = false;
    image.decoding = "async";
    const finish = async () => {
      if (settled) return;
      try {
        if (typeof image.decode === "function") await image.decode();
        settled = true;
        resolve(image);
      } catch (error) {
        settled = true;
        reject(error);
      }
    };
    image.addEventListener("load", finish, { once: true });
    image.addEventListener("error", () => reject(new Error(`Unable to load ${source}`)), { once: true });
    image.src = source;
    if (image.complete && image.naturalWidth > 0) void finish();
  });
}

async function loadMotionEngine() {
  const [{ gsap }, { CSSPlugin }, { default: ScrollTrigger }] = await Promise.all([
    import("./vendor/gsap/gsap-core.js"),
    import("./vendor/gsap/CSSPlugin.js"),
    import("./vendor/gsap/ScrollTrigger.js"),
  ]);

  gsap.registerPlugin(CSSPlugin, ScrollTrigger);
  return { gsap, ScrollTrigger };
}

export function initProductStory(documentRoot = document) {
  const view = documentRoot.defaultView || window;
  const track = documentRoot.querySelector("#story-track");
  const stage = documentRoot.querySelector("#story-stage");
  const art = documentRoot.querySelector("#story-art");
  const canvas = documentRoot.querySelector("#story-canvas");
  const staticImage = documentRoot.querySelector("#story-static-image");
  const caption = documentRoot.querySelector("#story-caption-stage");
  const progressFill = documentRoot.querySelector("#story-progress-fill");
  const steps = [...documentRoot.querySelectorAll("[data-story-step]")];
  const buttons = [...documentRoot.querySelectorAll("[data-story-go]")];

  if (!track || !stage || !art || !staticImage || !steps.length || buttons.length !== steps.length) return;

  const reducedMotion = view.matchMedia("(prefers-reduced-motion: reduce)");
  const staticPresentation = view.matchMedia("(max-width: 740px), (pointer: coarse)");
  const context = canvas?.getContext("2d", { alpha: true, desynchronized: true });
  let gsap;
  let scrollTrigger;
  let animatedStory = false;
  let activeStep = -1;
  let frameIndex = 0;
  const pointer = { x: 0, y: 0 };
  const imageSources = {};
  let resizeObserver;
  let canvasWidth = 0;
  let canvasHeight = 0;
  let pixelRatio = 1;

  function setActiveStep(index) {
    const nextIndex = clamp(index, 0, steps.length - 1);
    const changed = nextIndex !== activeStep;
    activeStep = nextIndex;

    steps.forEach((step, stepIndex) => {
      const active = stepIndex === nextIndex;
      step.hidden = animatedStory ? false : !active;
      if (active) step.removeAttribute("aria-hidden");
      else step.setAttribute("aria-hidden", "true");
    });

    buttons.forEach((button, buttonIndex) => {
      if (buttonIndex === nextIndex) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });

    stage.dataset.activeStep = String(nextIndex);
    if (caption) caption.textContent = `${String(nextIndex + 1).padStart(2, "0")} / ${String(steps.length).padStart(2, "0")}`;
    if (progressFill && changed) progressFill.style.width = `${((nextIndex + 0.5) / steps.length) * 100}%`;
  }

  function drawFrame(index = frameIndex) {
    if (!context || staticPresentation.matches || !canvasWidth || !canvasHeight) return;

    const theme = documentRoot.documentElement.dataset.theme === "dark" ? "dark" : "light";
    const image = imageSources[theme];
    if (!image || !image.naturalWidth) return;

    const frame = frameSequence[clamp(Math.round(index), 0, FRAME_COUNT - 1)];
    const pose = frame.camera;
    const baseSize = Math.min(canvasWidth * 0.78, canvasHeight * 0.98, 960);
    const parallaxScale = Math.min(canvasWidth, canvasHeight) * 0.016;

    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, canvasWidth, canvasHeight);
    context.save();
    context.translate(pose.x * canvasWidth + pointer.x * parallaxScale, pose.y * canvasHeight + pointer.y * parallaxScale);
    context.rotate(pose.rotate);
    context.scale(pose.scale, pose.scale);
    context.transform(1, 0, Math.sin((frame.index / (FRAME_COUNT - 1)) * Math.PI * 2) * 0.012, 1, 0, 0);
    context.drawImage(image, -baseSize / 2, -baseSize / 2, baseSize, baseSize);
    context.restore();
  }

  function resizeCanvas() {
    if (!canvas || !context || staticPresentation.matches) return;
    const bounds = canvas.getBoundingClientRect();
    pixelRatio = Math.min(view.devicePixelRatio || 1, 1.75);
    canvasWidth = bounds.width;
    canvasHeight = bounds.height;
    canvas.width = Math.max(1, Math.round(canvasWidth * pixelRatio));
    canvas.height = Math.max(1, Math.round(canvasHeight * pixelRatio));
    drawFrame();
  }

  function handleProgress(progress) {
    const stepIndex = Math.min(steps.length - 1, Math.floor(progress * steps.length));
    setActiveStep(stepIndex);
    if (progressFill) progressFill.style.width = `${progress * 100}%`;
  }

  function goToStep(index) {
    if (!gsap || !scrollTrigger || reducedMotion.matches) {
      setActiveStep(index);
      return;
    }

    const travel = Math.max(0, track.offsetHeight - stage.offsetHeight);
    const trackTop = view.scrollY + track.getBoundingClientRect().top;
    const progress = (index + 0.5) / steps.length;
    view.scrollTo({
      top: trackTop + travel * progress,
      behavior: "smooth",
    });
  }

  buttons.forEach((button) => {
    button.addEventListener("click", () => goToStep(Number(button.dataset.storyGo)));
  });

  setActiveStep(0);

  if (reducedMotion.matches) return;

  async function startCanvasStory() {
    const lightSource = staticImage.dataset.themeImageLight;
    const darkSource = staticImage.dataset.themeImageDark;
    if (!lightSource || !darkSource) return;

    try {
      [imageSources.light, imageSources.dark] = await Promise.all([
        loadImage(view, lightSource),
        loadImage(view, darkSource),
      ]);
      if (context && !staticPresentation.matches) art.classList.add("is-ready");
      resizeCanvas();

      const motionEngine = await loadMotionEngine();
      gsap = motionEngine.gsap;
      scrollTrigger = motionEngine.ScrollTrigger;
      animatedStory = true;
      setActiveStep(activeStep);
      gsap.set(steps, { autoAlpha: 0, y: 18 });
      gsap.set(steps[activeStep], { autoAlpha: 1, y: 0 });

      const playhead = { frame: 0 };
      const storyTimeline = gsap.timeline({
        scrollTrigger: {
          trigger: track,
          start: "top top",
          end: "bottom bottom",
          scrub: true,
          invalidateOnRefresh: true,
          onUpdate: (self) => handleProgress(self.progress),
          onRefresh: (self) => handleProgress(self.progress),
        },
      });
      storyTimeline.to(playhead, {
        frame: FRAME_COUNT - 1,
        duration: steps.length,
        ease: "none",
        onUpdate: () => {
          frameIndex = Math.round(playhead.frame);
          drawFrame(frameIndex);
        },
      }, 0);
      steps.slice(0, -1).forEach((step, index) => {
        const transition = index + 0.88;
        storyTimeline.to(step, { autoAlpha: 0, y: -12, duration: 0.15, ease: "none" }, transition);
        storyTimeline.fromTo(steps[index + 1], { autoAlpha: 0, y: 16 }, {
          autoAlpha: 1,
          y: 0,
          duration: 0.15,
          ease: "none",
        }, transition);
      });

      if (context && view.ResizeObserver) {
        resizeObserver = new view.ResizeObserver(resizeCanvas);
        resizeObserver.observe(stage);
      }

      const pointerXTo = context ? gsap.quickTo(pointer, "x", { duration: 0.36, ease: "power3.out", onUpdate: drawFrame }) : null;
      const pointerYTo = context ? gsap.quickTo(pointer, "y", { duration: 0.36, ease: "power3.out", onUpdate: drawFrame }) : null;
      stage.addEventListener("pointermove", (event) => {
        if (event.pointerType !== "mouse" || !pointerXTo || staticPresentation.matches) return;
        const bounds = stage.getBoundingClientRect();
        pointerXTo(((event.clientX - bounds.left) / bounds.width - 0.5) * 2);
        pointerYTo(((event.clientY - bounds.top) / bounds.height - 0.5) * 2);
      });
      stage.addEventListener("pointerleave", () => {
        pointerXTo?.(0);
        pointerYTo?.(0);
      });

      const themeObserver = new view.MutationObserver(() => drawFrame());
      themeObserver.observe(documentRoot.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
      view.addEventListener("resize", () => {
        if (context && !staticPresentation.matches) {
          art.classList.add("is-ready");
          resizeCanvas();
        } else {
          art.classList.remove("is-ready");
        }
        scrollTrigger.refresh();
      }, { passive: true });
      scrollTrigger.refresh();
    } catch {
      // Keep the responsive, high-resolution static image as the fallback.
    }
  }

  void startCanvasStory();
}
