(function () {
  function slideManually(carousel, direction) {
    var items = Array.prototype.slice.call(carousel.querySelectorAll(".carousel-item"));
    if (items.length < 2) return;

    var activeItem = carousel.querySelector(".carousel-item.active") || items[0];
    var activeIndex = items.indexOf(activeItem);
    var nextIndex = direction === "prev" ? activeIndex - 1 : activeIndex + 1;

    if (nextIndex < 0) nextIndex = items.length - 1;
    if (nextIndex >= items.length) nextIndex = 0;

    activeItem.classList.remove("active");
    items[nextIndex].classList.add("active");
  }

  document.addEventListener(
    "click",
    function (event) {
      var control = event.target.closest(".profile-carousel [data-slide]");
      if (!control) return;

      var carousel = document.querySelector(control.getAttribute("data-target") || control.getAttribute("href"));
      if (!carousel) return;

      event.preventDefault();
      event.stopPropagation();

      var direction = control.getAttribute("data-slide");

      if (window.jQuery && window.jQuery.fn && window.jQuery.fn.carousel) {
        window.jQuery(carousel).carousel(direction);
        return;
      }

      slideManually(carousel, direction);
    },
    true
  );
})();
