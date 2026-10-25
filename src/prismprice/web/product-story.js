const FRAME_COUNT = 180;

function clamp(value, minimum, maximum) {
  return Math.min(maximum, Math.max(minimum, value));
}

function interpolate(start, end, progress) {
  const eased = progress * progress * (3 - 2 * progress);
  return start + (end - start) * eased;
}

function roundedRect(context, x, y, width, height, radius, fill, stroke) {
  const r = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + r, y);
  context.arcTo(x + width, y, x + width, y + height, r);
  context.arcTo(x + width, y + height, x, y + height, r);
  context.arcTo(x, y + height, x, y, r);
  context.arcTo(x, y, x + width, y, r);
  context.closePath();
  if (fill) {
    context.fillStyle = fill;
    context.fill();
  }
  if (stroke) {
    context.strokeStyle = stroke;
    context.lineWidth = 1;
    context.stroke();
  }
}

function drawLabel(context, value, x, y, size, color, weight = 400, align = "left") {
  context.fillStyle = color;
  context.font = `£{weight} £{size}px -apple-system, BlinkMacSystemFont, "SF Pro Display", sans-serif`;
  context.textAlign = align;
  context.textBaseline = "alphabetic";
  context.fillText(value, x, y);
}

function drawLine(context, x1, y1, x2, y2, color, width = 1, dash = []) {
  context.beginPath();
  context.setLineDash(dash);
  context.moveTo(x1, y1);
  context.lineTo(x2, y2);
  context.strokeStyle = color;
  context.lineWidth = width;
  context.stroke();
  context.setLineDash([]);
}

function paletteFor(documentRoot) {
  if (documentRoot.documentElement.dataset.theme === "dark") {
    return {
      ground: "#17171b", panel: "#222226", soft: "#2b2b30", ink: "#f5f5f7",
      secondary: "#c2c2c7", muted: "#96969b", line: "#38383e", strong: "#4b4b52",
      blue: "#0a84ff", blueWash: "#172b40", green: "#8cc6a5", greenWash: "#25372d",
      grid: "#38383e", shadow: "rgba(0,0,0,.24)", avatar: "#393943",
    };
  }
  return {
    ground: "#f5f5f7", panel: "#ffffff", soft: "#f6f6f8", ink: "#1d1d1f",
    secondary: "#626269", muted: "#86868b", line: "#e6e6eb", strong: "#d2d2d7",
    blue: "#0071e3", blueWash: "#edf5ff", green: "#28714e", greenWash: "#edf6f0",
    grid: "#e6e6eb", shadow: "rgba(20,28,40,.11)", avatar: "#e8e5df",
  };
}

function drawDashboard(context, bounds, colors) {
  const { x, y, width, height } = bounds;
  const scale = width / 1000;
  const top = 56 * scale;
  const left = 34 * scale;
  const innerWidth = width - left * 2;
  const contentTop = y + top;

  context.save();
  context.shadowColor = colors.shadow;
  context.shadowBlur = 28 * scale;
  context.shadowOffsetY = 12 * scale;
  roundedRect(context, x, y, width, height, 18 * scale, colors.panel, colors.line);
  context.restore();
  roundedRect(context, x, y, width, height, 18 * scale, colors.panel, colors.line);
  context.save();
  context.beginPath();
  context.rect(x, y + 18 * scale, width, height - 18 * scale);
  context.clip();
  drawLine(context, x, y + top, x + width, y + top, colors.line);

  drawLabel(context, "PrismPrice", x + left, y + 35 * scale, 13 * scale, colors.ink, 650);
  drawLabel(context, "Pricing workspace", x + left + 92 * scale, y + 35 * scale, 10 * scale, colors.muted);
  roundedRect(context, x + width - 133 * scale, y + 17 * scale, 101 * scale, 23 * scale, 12 * scale, colors.soft, colors.line);
  drawLabel(context, "Scenario ready", x + width - 82 * scale, y + 32 * scale, 8 * scale, colors.secondary, 550, "center");
  context.beginPath();
  context.arc(x + width - 121 * scale, y + 28 * scale, 3 * scale, 0, Math.PI * 2);
  context.fillStyle = colors.green;
  context.fill();

  drawLabel(context, "PRODUCT  /  SKU-001", x + left, contentTop + 26 * scale, 8 * scale, colors.muted, 650);
  drawLabel(context, "Ceramic mug", x + left, contentTop + 54 * scale, 20 * scale, colors.ink, 620);
  drawLabel(context, "Demand, margin, and policy at a glance", x + width - left, contentTop + 51 * scale, 9 * scale, colors.muted, 400, "right");

  const cardY = contentTop + 72 * scale;
  const cardHeight = 136 * scale;
  roundedRect(context, x + left, cardY, innerWidth, cardHeight, 12 * scale, colors.panel, colors.line);
  drawLabel(context, "RECOMMENDED PRICE", x + left + 20 * scale, cardY + 24 * scale, 8 * scale, colors.muted, 650);
  drawLabel(context, "£32.40", x + left + 20 * scale, cardY + 66 * scale, 37 * scale, colors.ink, 630);
  drawLabel(context, "↑  8.0%", x + left + 21 * scale, cardY + 91 * scale, 10 * scale, colors.green, 620);
  drawLabel(context, "from £30.00", x + left + 69 * scale, cardY + 91 * scale, 9 * scale, colors.muted);
  drawLine(context, x + left + 167 * scale, cardY + 19 * scale, x + left + 167 * scale, cardY + 117 * scale, colors.line);
  drawLabel(context, "Expected demand", x + left + 193 * scale, cardY + 37 * scale, 9 * scale, colors.muted);
  drawLabel(context, "1,240", x + left + 193 * scale, cardY + 63 * scale, 19 * scale, colors.ink, 620);
  drawLabel(context, "units  ·  +8.4%", x + left + 193 * scale, cardY + 82 * scale, 8 * scale, colors.green, 500);
  drawLine(context, x + left + 343 * scale, cardY + 19 * scale, x + left + 343 * scale, cardY + 117 * scale, colors.line);
  drawLabel(context, "Gross margin", x + left + 369 * scale, cardY + 37 * scale, 9 * scale, colors.muted);
  drawLabel(context, "53.7%", x + left + 369 * scale, cardY + 63 * scale, 19 * scale, colors.ink, 620);
  drawLabel(context, "+3.7 pts vs. floor", x + left + 369 * scale, cardY + 82 * scale, 8 * scale, colors.green, 500);
  const badgeX = x + width - left - 119 * scale;
  roundedRect(context, badgeX, cardY + 47 * scale, 99 * scale, 26 * scale, 14 * scale, colors.greenWash);
  drawLabel(context, "✓  Within policy", badgeX + 49 * scale, cardY + 64 * scale, 8 * scale, colors.green, 600, "center");

  const analyticsY = cardY + cardHeight + 13 * scale;
  const analyticsHeight = Math.min(185 * scale, y + height - analyticsY - 18 * scale);
  const chartWidth = innerWidth * .65;
  const checkX = x + left + chartWidth + 13 * scale;
  const checkWidth = innerWidth - chartWidth - 13 * scale;
  roundedRect(context, x + left, analyticsY, chartWidth, analyticsHeight, 11 * scale, colors.panel, colors.line);
  roundedRect(context, checkX, analyticsY, checkWidth, analyticsHeight, 11 * scale, colors.panel, colors.line);
  drawLabel(context, "Demand by candidate price", x + left + 16 * scale, analyticsY + 22 * scale, 10 * scale, colors.ink, 580);
  drawLabel(context, "30-day forecast", x + left + chartWidth - 16 * scale, analyticsY + 22 * scale, 8 * scale, colors.muted, 400, "right");

  const graphX = x + left + 27 * scale;
  const graphY = analyticsY + 43 * scale;
  const graphW = chartWidth - 43 * scale;
  const graphH = analyticsHeight - 67 * scale;
  for (let i = 0; i < 3; i += 1) {
    const lineY = graphY + i * (graphH / 2);
    drawLine(context, graphX, lineY, graphX + graphW, lineY, colors.grid);
  }
  const points = [
    [0, .15], [.13, .21], [.27, .31], [.41, .38], [.55, .47], [.7, .53], [.83, .7], [1, .82],
  ];
  context.beginPath();
  points.forEach(([px, py], index) => {
    const pointX = graphX + px * graphW;
    const pointY = graphY + py * graphH;
    if (index === 0) context.moveTo(pointX, pointY);
    else context.lineTo(pointX, pointY);
  });
  context.lineTo(graphX + graphW, graphY + graphH);
  context.lineTo(graphX, graphY + graphH);
  context.closePath();
  context.fillStyle = colors.blueWash;
  context.globalAlpha = .8;
  context.fill();
  context.globalAlpha = 1;
  context.beginPath();
  points.forEach(([px, py], index) => {
    const pointX = graphX + px * graphW;
    const pointY = graphY + py * graphH;
    if (index === 0) context.moveTo(pointX, pointY);
    else context.lineTo(pointX, pointY);
  });
  context.strokeStyle = colors.blue;
  context.lineWidth = 2 * scale;
  context.lineJoin = "round";
  context.stroke();
  const selectedX = graphX + graphW * .7;
  const selectedY = graphY + graphH * .53;
  drawLine(context, selectedX, graphY, selectedX, graphY + graphH, colors.strong, 1, [3 * scale, 4 * scale]);
  context.beginPath();
  context.arc(selectedX, selectedY, 5 * scale, 0, Math.PI * 2);
  context.fillStyle = colors.panel;
  context.fill();
  context.strokeStyle = colors.blue;
  context.lineWidth = 2 * scale;
  context.stroke();

  drawLabel(context, "Policy checks", checkX + 15 * scale, analyticsY + 22 * scale, 10 * scale, colors.ink, 580);
  const policyRows = ["Margin floor", "Price movement", "Promotion rules"];
  policyRows.forEach((label, index) => {
    const rowY = analyticsY + (53 + index * 34) * scale;
    drawLine(context, checkX + 13 * scale, rowY - 14 * scale, checkX + checkWidth - 13 * scale, rowY - 14 * scale, colors.line);
    context.beginPath();
    context.arc(checkX + 21 * scale, rowY - 2 * scale, 6 * scale, 0, Math.PI * 2);
    context.fillStyle = colors.greenWash;
    context.fill();
    drawLabel(context, "✓", checkX + 21 * scale, rowY + 1 * scale, 8 * scale, colors.green, 650, "center");
    drawLabel(context, label, checkX + 34 * scale, rowY + 1 * scale, 8 * scale, colors.secondary);
    drawLabel(context, "Passed", checkX + checkWidth - 13 * scale, rowY + 1 * scale, 8 * scale, colors.green, 550, "right");
  });

  const footerY = analyticsY + analyticsHeight + 13 * scale;
  const footerHeight = Math.min(112 * scale, y + height - footerY - 14 * scale);
  const ladderWidth = innerWidth * .62;
  const recordX = x + left + ladderWidth + 13 * scale;
  const recordWidth = innerWidth - ladderWidth - 13 * scale;
  roundedRect(context, x + left, footerY, ladderWidth, footerHeight, 11 * scale, colors.panel, colors.line);
  roundedRect(context, recordX, footerY, recordWidth, footerHeight, 11 * scale, colors.panel, colors.line);
  drawLabel(context, "Candidate prices", x + left + 15 * scale, footerY + 19 * scale, 9 * scale, colors.ink, 580);
  drawLabel(context, "DEMAND", x + left + ladderWidth - 105 * scale, footerY + 19 * scale, 6 * scale, colors.muted, 650);
  drawLabel(context, "MARGIN", x + left + ladderWidth - 47 * scale, footerY + 19 * scale, 6 * scale, colors.muted, 650);
  const candidates = [
    ["£28.50", .92, "46%"], ["£30.00", .8, "50%"], ["£32.40", .68, "54%"], ["£35.00", .51, "58%"],
  ];
  candidates.forEach(([price, demand, margin], index) => {
    const rowY = footerY + (38 + index * 17) * scale;
    const selected = index === 2;
    if (selected) roundedRect(context, x + left + 8 * scale, rowY - 12 * scale, ladderWidth - 16 * scale, 15 * scale, 5 * scale, colors.blueWash);
    drawLabel(context, price, x + left + 15 * scale, rowY, 7 * scale, selected ? colors.blue : colors.secondary, selected ? 620 : 450);
    roundedRect(context, x + left + 82 * scale, rowY - 6 * scale, Math.max(22, ladderWidth - 161 * scale) * demand, 4 * scale, 3 * scale, selected ? colors.blue : colors.strong);
    drawLabel(context, margin, x + left + ladderWidth - 23 * scale, rowY, 7 * scale, colors.secondary, 500, "right");
  });

  drawLabel(context, "Decision record", recordX + 15 * scale, footerY + 20 * scale, 9 * scale, colors.ink, 580);
  drawLine(context, recordX + 13 * scale, footerY + 29 * scale, recordX + recordWidth - 13 * scale, footerY + 29 * scale, colors.line);
  drawLabel(context, "MODEL VERSION", recordX + 15 * scale, footerY + 46 * scale, 6 * scale, colors.muted, 650);
  drawLabel(context, "v2.4", recordX + recordWidth - 15 * scale, footerY + 46 * scale, 7 * scale, colors.secondary, 550, "right");
  drawLabel(context, "REASON CODES", recordX + 15 * scale, footerY + 64 * scale, 6 * scale, colors.muted, 650);
  drawLabel(context, "4 recorded", recordX + recordWidth - 15 * scale, footerY + 64 * scale, 7 * scale, colors.secondary, 550, "right");
  roundedRect(context, recordX + 14 * scale, footerY + 77 * scale, recordWidth - 28 * scale, 20 * scale, 6 * scale, colors.soft);
  drawLabel(context, "Review before publishing", recordX + recordWidth / 2, footerY + 91 * scale, 7 * scale, colors.secondary, 500, "center");
  context.restore();
}

function cameraAt(frame, width, height, board) {
  const progress = frame / (FRAME_COUNT - 1);
  const segmentCount = 4;
  const segment = Math.min(segmentCount - 1, Math.floor(progress * segmentCount));
  const local = progress >= 1 ? 1 : progress * segmentCount - segment;
  const targets = [
    { scale: .92, x: board.x + board.width * .53, y: board.y + board.height * .5 },
    { scale: 1.12, x: board.x + board.width * .39, y: board.y + board.height * .34 },
    { scale: 1.27, x: board.x + board.width * .39, y: board.y + board.height * .63 },
    { scale: 1.55, x: board.x + board.width * .79, y: board.y + board.height * .62 },
    { scale: 1.78, x: board.x + board.width * .77, y: board.y + board.height * .84 },
  ];
  const start = targets[segment];
  const end = targets[segment + 1];
  return {
    scale: interpolate(start.scale, end.scale, local),
    x: interpolate(start.x, end.x, local),
    y: interpolate(start.y, end.y, local),
    width,
    height,
  };
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
  const caption = documentRoot.querySelector("#story-caption-stage");
  const progressFill = documentRoot.querySelector("#story-progress-fill");
  const steps = [...documentRoot.querySelectorAll("[data-story-step]")];
  const buttons = [...documentRoot.querySelectorAll("[data-story-go]")];
  if (!track || !stage || !art || !steps.length || buttons.length !== steps.length) return;

  const reducedMotion = view.matchMedia("(prefers-reduced-motion: reduce)");
  const staticPresentation = view.matchMedia("(max-width: 740px), (pointer: coarse)");
  const context = canvas?.getContext("2d", { alpha: false, desynchronized: true });
  let gsap;
  let scrollTrigger;
  let animatedStory = false;
  let activeStep = -1;
  let frameIndex = 0;
  let canvasWidth = 0;
  let canvasHeight = 0;
  let pixelRatio = 1;
  const pointer = { x: 0, y: 0 };

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
    if (caption) caption.textContent = `£{String(nextIndex + 1).padStart(2, "0")} / £{String(steps.length).padStart(2, "0")}`;
    if (progressFill && changed) progressFill.style.width = `£{((nextIndex + 0.5) / steps.length) * 100}%`;
  }

  function drawFrame(index = frameIndex) {
    if (!context || !canvasWidth || !canvasHeight || staticPresentation.matches) return;
    const colors = paletteFor(documentRoot);
    const boardWidth = Math.min(canvasWidth * .72, 1000);
    const boardHeight = Math.min(canvasHeight * .78, boardWidth / 1.43);
    const boardX = Math.max(canvasWidth * .29, (canvasWidth - boardWidth) * .78);
    const boardY = (canvasHeight - boardHeight) / 2;
    const board = { x: boardX, y: boardY, width: boardWidth, height: boardHeight };
    const camera = cameraAt(clamp(Math.round(index), 0, FRAME_COUNT - 1), canvasWidth, canvasHeight, board);
    const parallax = Math.min(canvasWidth, canvasHeight) * .008;

    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.fillStyle = colors.ground;
    context.fillRect(0, 0, canvasWidth, canvasHeight);
    context.setTransform(
      pixelRatio * camera.scale,
      0,
      0,
      pixelRatio * camera.scale,
      pixelRatio * (canvasWidth / 2 - camera.x * camera.scale + pointer.x * parallax),
      pixelRatio * (canvasHeight / 2 - camera.y * camera.scale + pointer.y * parallax),
    );
    drawDashboard(context, board, colors);
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
    if (progressFill) progressFill.style.width = `£{progress * 100}%`;
  }

  function goToStep(index) {
    if (!gsap || !scrollTrigger || reducedMotion.matches || staticPresentation.matches) {
      setActiveStep(index);
      return;
    }
    const travel = Math.max(0, track.offsetHeight - stage.offsetHeight);
    const trackTop = view.scrollY + track.getBoundingClientRect().top;
    const progress = (index + .5) / steps.length;
    view.scrollTo({ top: trackTop + travel * progress, behavior: "smooth" });
  }

  buttons.forEach((button) => button.addEventListener("click", () => goToStep(Number(button.dataset.storyGo))));
  setActiveStep(0);
  if (reducedMotion.matches || staticPresentation.matches) {
    track.classList.add("is-static");
    return;
  }

  function showStaticStory() {
    if (gsap) gsap.set(steps, { clearProps: "all" });
    animatedStory = false;
    setActiveStep(activeStep);
    track.classList.add("is-static");
    if (art) art.classList.remove("is-ready");
  }

  async function startCanvasStory() {
    try {
      const motionEngine = await loadMotionEngine();
      gsap = motionEngine.gsap;
      scrollTrigger = motionEngine.ScrollTrigger;
      if (!context) {
        showStaticStory();
        return;
      }

      resizeCanvas();
      art.classList.add("is-ready");
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
        const transition = index + .88;
        storyTimeline.to(step, { autoAlpha: 0, y: -12, duration: .15, ease: "none" }, transition);
        storyTimeline.fromTo(steps[index + 1], { autoAlpha: 0, y: 16 }, {
          autoAlpha: 1,
          y: 0,
          duration: .15,
          ease: "none",
        }, transition);
      });

      if (view.ResizeObserver) {
        const resizeObserver = new view.ResizeObserver(resizeCanvas);
        resizeObserver.observe(stage);
      }
      const pointerXTo = gsap.quickTo(pointer, "x", { duration: .36, ease: "power3.out", onUpdate: drawFrame });
      const pointerYTo = gsap.quickTo(pointer, "y", { duration: .36, ease: "power3.out", onUpdate: drawFrame });
      stage.addEventListener("pointermove", (event) => {
        if (event.pointerType !== "mouse" || staticPresentation.matches) return;
        const bounds = stage.getBoundingClientRect();
        pointerXTo(((event.clientX - bounds.left) / bounds.width - .5) * 2);
        pointerYTo(((event.clientY - bounds.top) / bounds.height - .5) * 2);
      });
      stage.addEventListener("pointerleave", () => {
        pointerXTo(0);
        pointerYTo(0);
      });

      const themeObserver = new view.MutationObserver(() => drawFrame());
      themeObserver.observe(documentRoot.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
      view.addEventListener("resize", () => {
        resizeCanvas();
        scrollTrigger.refresh();
      }, { passive: true });
      scrollTrigger.refresh();
    } catch {
      // The HTML dashboard remains visible if canvas animation is unavailable.
      showStaticStory();
    }
  }

  void startCanvasStory();
}
