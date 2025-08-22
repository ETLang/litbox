var MobileDetector = {
  IsMobile: function () {
    return /Mobi|Android/i.test(navigator.userAgent);
  }
};

mergeInto(LibraryManager.library, MobileDetector);  