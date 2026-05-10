/**
 * Roadmap stepper — highlights the active section as the user navigates.
 * Reveal.js fires a `slidechanged` event we hook into.
 */
(function () {
  const ROADMAP_BY_SECTION = {
    1: "shape",
    2: "lifecycle",
    3: "data-auth",
    4: "api-patterns",
    5: "test-deploy",
    6: "production",
  };

  function setActiveRoadmap(stepName) {
    document.querySelectorAll(".roadmap-item").forEach((el) => {
      el.classList.toggle("active", el.dataset.step === stepName);
    });
  }

  function onSlideChanged(event) {
    const slide = event.currentSlide;
    if (!slide) return;
    const stepAttr = slide.dataset.roadmap;
    if (stepAttr) setActiveRoadmap(stepAttr);
  }

  document.addEventListener("DOMContentLoaded", () => {
    if (typeof Reveal === "undefined") return;
    Reveal.on("slidechanged", onSlideChanged);
    // Initialise to the first slide's roadmap (or default).
    const first = document.querySelector(".reveal .slides section");
    if (first && first.dataset.roadmap) setActiveRoadmap(first.dataset.roadmap);
  });

  // Suppress the unused variable lint; ROADMAP_BY_SECTION reserved for
  // future use (jump-to-section navigation).
  void ROADMAP_BY_SECTION;
})();
