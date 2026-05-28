/**
 * Surya Prakash - AI/ML Portfolio Script
 * Features:
 *  - Canvas Neural Network Particle System (cursor attraction, interactive connection links)
 *  - Animated Skill Bars (triggers scroll-reveal progress expansion & numerical counter increments)
 *  - Typewriter/Fade Rotation for Hero Title subtitles
 *  - Interactive print trigger for resume sheet (window.print() with custom layout styling)
 *  - Contact Form validation, drafting in LocalStorage, and Glassmorphic Toast Notifications
 *  - Intersection Observer for scroll highlighting in Navigation Links
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Lucide icons
  lucide.createIcons();

  // Initialize Canvas Neural Background
  initNeuralBackground();

  // Initialize Animated Skill Bars
  initSkillBars();

  // Initialize Text Rotator
  initTextRotator();

  // Initialize Mobile Menu
  initMobileMenu();

  // Initialize Resume PDF/Print Trigger
  initResumePrint();

  // Initialize Active Nav Links Highlight & Smooth Scroll
  initNavigationHighlight();

  // Initialize Contact Form
  initContactForm();
});

/* ==========================================================================
   1. Canvas Neural Network Particle Background
   ========================================================================== */
function initNeuralBackground() {
  const canvas = document.getElementById('neural-canvas');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  let animationId = null;
  let particles = [];
  const maxParticles = window.innerWidth < 768 ? 40 : 100;
  const connectionDistance = 120;
  const mouse = { x: null, y: null, radius: 150 };

  // Track cursor position
  window.addEventListener('mousemove', (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
  });

  window.addEventListener('mouseleave', () => {
    mouse.x = null;
    mouse.y = null;
  });

  // Adjust canvas size
  function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  // Particle Class Definition
  class Particle {
    constructor() {
      this.x = Math.random() * canvas.width;
      this.y = Math.random() * canvas.height;
      this.vx = (Math.random() - 0.5) * 0.4;
      this.vy = (Math.random() - 0.5) * 0.4;
      this.size = Math.random() * 2 + 1;
      this.color = 'rgba(99, 102, 241, 0.4)'; // Primary indigo hue
    }

    update() {
      // Boundaries bounce
      if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
      if (this.y < 0 || this.y > canvas.height) this.vy *= -1;

      // Update positions
      this.x += this.vx;
      this.y += this.vy;

      // Mouse interactive influence (slight pull/push)
      if (mouse.x !== null && mouse.y !== null) {
        const dx = mouse.x - this.x;
        const dy = mouse.y - this.y;
        const distance = Math.hypot(dx, dy);

        if (distance < mouse.radius) {
          // Attract particles slightly to the cursor
          const force = (mouse.radius - distance) / mouse.radius;
          this.x += (dx / distance) * force * 0.5;
          this.y += (dy / distance) * force * 0.5;
        }
      }
    }

    draw() {
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fillStyle = this.color;
      ctx.fill();
    }
  }

  // Populate system
  for (let i = 0; i < maxParticles; i++) {
    particles.push(new Particle());
  }

  // Draw connections between nodes
  function connect() {
    for (let a = 0; a < particles.length; a++) {
      for (let b = a + 1; b < particles.length; b++) {
        const dx = particles[a].x - particles[b].x;
        const dy = particles[a].y - particles[b].y;
        const distance = Math.hypot(dx, dy);

        if (distance < connectionDistance) {
          // Fade connection lines out as nodes drift apart
          const opacity = (connectionDistance - distance) / connectionDistance * 0.15;
          ctx.strokeStyle = `rgba(99, 102, 241, ${opacity})`;
          ctx.lineWidth = 0.8;
          ctx.beginPath();
          ctx.moveTo(particles[a].x, particles[a].y);
          ctx.lineTo(particles[b].x, particles[b].y);
          ctx.stroke();
        }
      }

      // Connect nodes to mouse cursor
      if (mouse.x !== null && mouse.y !== null) {
        const dx = particles[a].x - mouse.x;
        const dy = particles[a].y - mouse.y;
        const distance = Math.hypot(dx, dy);

        if (distance < mouse.radius) {
          const opacity = (mouse.radius - distance) / mouse.radius * 0.25;
          ctx.strokeStyle = `rgba(6, 182, 212, ${opacity})`; // Accent Cyan color for cursor links
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(particles[a].x, particles[a].y);
          ctx.lineTo(mouse.x, mouse.y);
          ctx.stroke();
        }
      }
    }
  }

  // Animation loop
  function animate() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    particles.forEach(p => {
      p.update();
      p.draw();
    });

    connect();
    animationId = requestAnimationFrame(animate);
  }

  animate();
}

/* ==========================================================================
   2. Skill Bars Intersection Observer & Number Counter Animation
   ========================================================================== */
function initSkillBars() {
  const skillBoxes = document.querySelectorAll('.skill-box');
  if (skillBoxes.length === 0) return;

  const observerOptions = {
    threshold: 0.15,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver((entries, self) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const skillWrappers = entry.target.querySelectorAll('.skill-bar-wrapper');

        skillWrappers.forEach(wrapper => {
          const progress = wrapper.querySelector('.skill-progress');
          const valueText = wrapper.querySelector('[data-target]');

          if (progress && valueText) {
            const targetVal = parseInt(valueText.getAttribute('data-target'), 10);

            // Set width of bar
            progress.style.width = `${targetVal}%`;

            // Numerical Counter increment animation
            animateCountUp(valueText, targetVal);
          }
        });

        // Unobserve once triggered to avoid layout stutter
        self.unobserve(entry.target);
      }
    });
  }, observerOptions);

  skillBoxes.forEach(box => observer.observe(box));
}

// Numerical count up helper
function animateCountUp(element, target) {
  let current = 0;
  const duration = 1000; // 1 second
  const stepTime = Math.max(Math.floor(duration / target), 10);

  const timer = setInterval(() => {
    current += 1;
    element.textContent = `${current}%`;

    if (current >= target) {
      element.textContent = `${target}%`;
      clearInterval(timer);
    }
  }, stepTime);
}

/* ==========================================================================
   3. Typewriter / Rotate Subtitle Effects
   ========================================================================== */
function initTextRotator() {
  const element = document.getElementById('hero-title');
  if (!element) return;

  const titles = [
    "AI/ML Developer",
    "Computer Science Student",
    "Deep Learning Enthusiast"
  ];
  let titleIndex = 0;
  let charIndex = 0;
  let isDeleting = false;
  let typeSpeed = 100;

  function type() {
    const currentTitle = titles[titleIndex];

    if (isDeleting) {
      // Deleting character
      element.textContent = currentTitle.substring(0, charIndex - 1);
      charIndex--;
      typeSpeed = 40; // delete faster
    } else {
      // Adding character
      element.textContent = currentTitle.substring(0, charIndex + 1);
      charIndex++;
      typeSpeed = 80;
    }

    // Finished typing full word
    if (!isDeleting && charIndex === currentTitle.length) {
      isDeleting = true;
      typeSpeed = 2000; // Pause at full word
    } else if (isDeleting && charIndex === 0) {
      isDeleting = false;
      titleIndex = (titleIndex + 1) % titles.length;
      typeSpeed = 300; // Short break before starting next word
    }

    setTimeout(type, typeSpeed);
  }

  // Clear static content and start typing
  element.textContent = '';
  setTimeout(type, 800);
}

/* ==========================================================================
   4. Mobile Menu Navigation
   ========================================================================== */
function initMobileMenu() {
  const btn = document.getElementById('mobile-menu-btn');
  const menu = document.getElementById('mobile-menu');

  if (btn && menu) {
    btn.addEventListener('click', () => {
      const isExpanded = menu.classList.contains('hidden');

      menu.classList.toggle('hidden');

      const icon = btn.querySelector('i');
      if (icon) {
        if (isExpanded) {
          icon.setAttribute('data-lucide', 'x');
        } else {
          icon.setAttribute('data-lucide', 'menu');
        }
        lucide.createIcons();
      }
    });

    // Close menu when clicking on a link
    menu.querySelectorAll('a').forEach(link => {
      link.addEventListener('click', () => {
        menu.classList.add('hidden');
        const icon = btn.querySelector('i');
        if (icon) {
          icon.setAttribute('data-lucide', 'menu');
          lucide.createIcons();
        }
      });
    });
  }
}

/* ==========================================================================
   5. Resume Print Layout PDF Download Action
   ========================================================================== */
function initResumePrint() {
  const printBtn = document.getElementById('btn-print-cv');
  if (printBtn) {
    printBtn.addEventListener('click', () => {
      // Trigger native print process
      window.print();
    });
  }
}

/* ==========================================================================
   6. Navigation Active Links Highlight
   ========================================================================== */
function initNavigationHighlight() {
  const sections = document.querySelectorAll('section[id]');
  const navLinks = document.querySelectorAll('nav a[href^="#"]');

  // IntersectionObserver to highlight navigation links based on viewport dominance
  const observerOptions = {
    root: null,
    rootMargin: '-30% 0px -65% 0px', // Target active area in upper-middle viewport
    threshold: 0
  };

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.getAttribute('id');
        navLinks.forEach(link => {
          if (link.getAttribute('href') === `#${id}`) {
            link.classList.add('text-primary-500', 'font-semibold');
          } else {
            link.classList.remove('text-primary-500', 'font-semibold');
          }
        });
      }
    });
  }, observerOptions);

  sections.forEach(section => observer.observe(section));

  // Performant Header Background intensity modifier on scroll (No layout queries)
  const nav = document.querySelector('nav');
  if (nav) {
    function toggleHeaderBackground() {
      if (window.scrollY > 50) {
        nav.classList.add('shadow-lg', 'bg-slate-950/90', 'border-slate-800/90');
      } else {
        nav.classList.remove('shadow-lg', 'bg-slate-950/90', 'border-slate-800/90');
      }
    }
    // Passive scroll listener ensures zero thread blockage for scroll behaviors
    window.addEventListener('scroll', toggleHeaderBackground, { passive: true });
    toggleHeaderBackground(); // Run initially
  }
}

/* ==========================================================================
   7. Contact Form Management (Draft preservation, Toast Notifiers)
   ========================================================================== */
function initContactForm() {
  const form = document.getElementById('contact-form');
  if (!form) return;

  const nameInput = document.getElementById('name');
  const emailInput = document.getElementById('email');
  const subjectInput = document.getElementById('subject');
  const messageInput = document.getElementById('message');

  // Load drafts if available in LocalStorage
  if (localStorage.getItem('contact_name')) nameInput.value = localStorage.getItem('contact_name');
  if (localStorage.getItem('contact_email')) emailInput.value = localStorage.getItem('contact_email');
  if (localStorage.getItem('contact_subject')) subjectInput.value = localStorage.getItem('contact_subject');
  if (localStorage.getItem('contact_message')) messageInput.value = localStorage.getItem('contact_message');

  // Listen for typing events to preserve drafts
  const inputs = [nameInput, emailInput, subjectInput, messageInput];
  inputs.forEach(input => {
    input.addEventListener('input', (e) => {
      localStorage.setItem(`contact_${e.target.id}`, e.target.value);
    });
  });

  // Handle Form Submission
  form.addEventListener('submit', async (e) => {
    e.preventDefault();

    const name = nameInput.value.trim();
    const email = emailInput.value.trim();
    const subject = subjectInput.value.trim();
    const message = messageInput.value.trim();

    // Verify fields
    if (!name || !email || !subject || !message) {
      showToast('All form fields are required!', 'error');
      return;
    }

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      showToast('Please enter a valid email address.', 'error');
      return;
    }

    // Capture submit button to show loading spinner state
    const submitBtn = form.querySelector('button[type="submit"]');
    const submitBtnText = submitBtn.querySelector('span');
    const originalText = submitBtnText.textContent;

    try {
      // Toggle button disabled state and show loader
      submitBtn.disabled = true;
      submitBtnText.textContent = 'Sending Message...';
      submitBtn.classList.add('opacity-80', 'cursor-not-allowed');

      // Access key setup for Web3Forms
      const web3FormsKey = '615c35c3-4390-4b69-b85f-0e37a7d5621d';

      // Real-world secure API post delivery using FormData to capture attributes cleanly
      const formData = new FormData(form);
      formData.append("access_key", web3FormsKey);

      const response = await fetch('https://api.web3forms.com/submit', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.message || 'Network request failed');
      }

      // Success visual feedback
      showToast(`Thank you, ${name}! Your message has been sent successfully.`, 'success');

      // Reset Form & Clear Local Drafts
      form.reset();
      inputs.forEach(input => localStorage.removeItem(`contact_${input.id}`));

    } catch (error) {
      console.error('Contact Form Delivery Error:', error);
      showToast('Failed to deliver message. Please try again later.', 'error');
    } finally {
      // Restore button state
      submitBtn.disabled = false;
      submitBtnText.textContent = originalText;
      submitBtn.classList.remove('opacity-80', 'cursor-not-allowed');
    }
  });
}

/* ==========================================================================
   8. Glassmorphic Notification Toast System
   ========================================================================== */
function showToast(message, type = 'success') {
  // Remove duplicate notifications if any
  const oldToast = document.querySelector('.toast-notification');
  if (oldToast) oldToast.remove();

  // Scaffold notification box
  const toast = document.createElement('div');
  toast.className = `toast-notification fixed bottom-6 right-6 z-50 px-6 py-4 rounded-xl shadow-2xl glass-card flex items-center gap-3 border transition-all duration-500 translate-y-10 opacity-0 max-w-sm`;

  if (type === 'success') {
    toast.classList.add('border-primary-500/35', 'text-slate-100');
    toast.innerHTML = `
      <div class="w-8 h-8 rounded-lg bg-primary-500/10 flex items-center justify-center flex-shrink-0 text-primary-400">
        <i data-lucide="check-circle" class="w-5 h-5"></i>
      </div>
      <div class="text-sm font-medium leading-tight">${message}</div>
    `;
  } else {
    toast.classList.add('border-accent-pink/35', 'text-slate-100');
    toast.innerHTML = `
      <div class="w-8 h-8 rounded-lg bg-accent-pink/10 flex items-center justify-center flex-shrink-0 text-accent-pink">
        <i data-lucide="alert-circle" class="w-5 h-5"></i>
      </div>
      <div class="text-sm font-medium leading-tight">${message}</div>
    `;
  }

  document.body.appendChild(toast);
  lucide.createIcons();

  // Slide in & Fade in
  setTimeout(() => {
    toast.classList.remove('translate-y-10', 'opacity-0');
  }, 50);

  // Auto clean up after 4.5 seconds
  setTimeout(() => {
    toast.classList.add('translate-y-10', 'opacity-0');
    setTimeout(() => toast.remove(), 550);
  }, 4500);
}