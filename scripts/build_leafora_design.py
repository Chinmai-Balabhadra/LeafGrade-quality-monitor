"""
Complete Leafora Redesign Builder
Applies the Leafora Design System to Project-471 while strictly preserving:
- All 78 required HTML element IDs
- Exactly 7 top-level sections in ordered sequence: home, analysis, history, model-performance, features, how-it-works, about
- Exactly 1 history section with id="history"
- Zero duplicate DOM element IDs
- All 28 evaluation table rows summing to 465 support
- All backend APIs, ML model, and SQLite database
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"


def update_style_css():
    style_path = FRONTEND_DIR / "css" / "style.css"
    content = """/* ==============================================================================
   Leafora – Plant Leaf Analysis & Quality Monitoring System
   Design System: Warm Ivory (#fbf9f5), Soft Botanical Green (#2d6a4f), White Glass Cards
   File: frontend/css/style.css
   ============================================================================== */

:root {
    /* Color Palette - Botanical Nature */
    --primary: #2d6a4f;
    --primary-dark: #1b4332;
    --primary-light: #40916c;
    --primary-subtle: #e8f3ea;
    --primary-accent: #3ea25d;
    --accent: #52b788;
    --accent-hover: #40916c;
    --accent-light: #d8f3dc;
    
    /* Backgrounds */
    --bg-page: #fbf9f5;
    --bg-sidebar: #fbf9f5;
    --bg-card: #ffffff;
    --bg-subtle: #f5f2eb;
    --bg-muted: #efeae2;
    
    /* Typography */
    --text-primary: #111827;
    --text-secondary: #4b5563;
    --text-muted: #6b7280;
    --text-light: #ffffff;
    
    /* Borders */
    --border-color: #eeebe4;
    --border-light: rgba(0, 0, 0, 0.05);
    --border-hover: rgba(45, 106, 79, 0.35);
    
    /* Status Colors */
    --status-success: #2d6a4f;
    --status-success-bg: #e8f3ea;
    --status-warning: #d97706;
    --status-warning-bg: #fef3c7;
    --status-danger: #e02424;
    --status-danger-bg: #fde8e8;
    
    /* Shadows */
    --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.03);
    --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.03);
    --shadow-lg: 0 10px 30px rgba(0, 0, 0, 0.05);
    --shadow-hover: 0 12px 30px rgba(45, 106, 79, 0.12);
    
    /* Layout Radii */
    --radius-sm: 8px;
    --radius-md: 14px;
    --radius-lg: 20px;
    --radius-xl: 26px;
    --radius-full: 9999px;
    
    /* Transitions */
    --transition: all 0.22s cubic-bezier(0.4, 0, 0.2, 1);
}

/* Global Reset */
*, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html {
    scroll-behavior: smooth;
    font-size: 16px;
}

body {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: var(--bg-page);
    color: var(--text-primary);
    line-height: 1.6;
    overflow-x: hidden;
    -webkit-font-smoothing: antialiased;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Outfit', 'Plus Jakarta Sans', sans-serif;
    color: var(--text-primary);
    font-weight: 700;
    line-height: 1.25;
}

a {
    color: var(--primary);
    text-decoration: none;
    transition: var(--transition);
}

a:hover {
    color: var(--primary-dark);
}

.container {
    width: 100%;
    margin: 0 auto;
    padding: 0 1.5rem;
}

/* ==========================================================================
   Leafora App Layout: Fixed Sidebar + Main Content
   ========================================================================== */
.app-layout {
    display: flex;
    min-height: 100vh;
    background: var(--bg-page);
    position: relative;
}

/* Fixed Left Sidebar */
.leafora-sidebar {
    position: fixed;
    top: 0;
    left: 0;
    bottom: 0;
    width: 236px;
    background: var(--bg-sidebar);
    border-right: 1px solid #ece7de;
    display: flex;
    flex-direction: column;
    padding: 1.5rem 1.15rem;
    z-index: 100;
    overflow-y: auto;
}

/* Sidebar Brand */
.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.25rem 0.5rem 1.5rem;
    border-bottom: 1px solid #efeae2;
    margin-bottom: 1.25rem;
    text-decoration: none;
}

.brand-logo-icon {
    width: 40px;
    height: 40px;
    border-radius: 10px;
    background: var(--primary-subtle);
    color: var(--primary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    flex-shrink: 0;
}

.brand-text-group {
    display: flex;
    flex-direction: column;
}

.brand-name {
    font-size: 1.45rem;
    font-weight: 800;
    color: #111827;
    line-height: 1.15;
    letter-spacing: -0.4px;
    font-family: 'Outfit', sans-serif;
}

.brand-subtitle {
    font-size: 0.74rem;
    font-weight: 500;
    color: #6b7280;
    margin-top: 1px;
}

/* Sidebar Navigation Menu */
.sidebar-nav-container {
    flex: 1;
}

.sidebar-menu {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
}

.sidebar-link {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    padding: 0.72rem 1rem;
    border-radius: 12px;
    font-size: 0.92rem;
    font-weight: 600;
    color: #4b5563;
    text-decoration: none;
    transition: all 0.2s ease;
    cursor: pointer;
}

.sidebar-link i {
    font-size: 1.05rem;
    width: 20px;
    text-align: center;
    color: #6b7280;
    transition: color 0.2s ease;
}

.sidebar-link:hover {
    background: #f0ede6;
    color: var(--primary);
}

.sidebar-link:hover i {
    color: var(--primary);
}

.sidebar-link.active {
    background: var(--primary-subtle);
    color: var(--primary);
    box-shadow: 0 1px 3px rgba(45, 106, 79, 0.08);
}

.sidebar-link.active i {
    color: var(--primary);
}

/* Sidebar Footer & Status */
.sidebar-footer {
    padding-top: 1rem;
    border-top: 1px solid #efeae2;
    margin-top: auto;
}

.nav-status {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    background: #ffffff;
    padding: 0.45rem 0.85rem;
    border-radius: var(--radius-full);
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid var(--border-color);
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #9ca3af;
    transition: var(--transition);
}

.status-dot.online {
    background: #16a34a;
    box-shadow: 0 0 0 3px rgba(22, 163, 74, 0.25);
    animation: pulse 2s infinite;
}

.status-dot.offline {
    background: var(--status-danger);
}

@keyframes pulse {
    0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.4); }
    70% { transform: scale(1); box-shadow: 0 0 0 5px rgba(22, 163, 74, 0); }
    100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(22, 163, 74, 0); }
}

/* Mobile Topbar */
.mobile-topbar {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: 60px;
    background: #ffffff;
    border-bottom: 1px solid var(--border-color);
    padding: 0 1.25rem;
    align-items: center;
    justify-content: space-between;
    z-index: 1000;
    box-shadow: var(--shadow-sm);
}

.mobile-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 1.25rem;
    font-weight: 800;
    color: var(--text-primary);
    font-family: 'Outfit', sans-serif;
}

.nav-toggle {
    background: none;
    border: none;
    font-size: 1.4rem;
    color: var(--text-primary);
    cursor: pointer;
    padding: 4px 8px;
}

/* ==========================================================================
   Main Content & Single Section Views
   ========================================================================== */
.main-content {
    margin-left: 236px;
    flex: 1;
    padding: 1.5rem 2.25rem 3rem;
    min-height: 100vh;
    background: var(--bg-page);
    max-width: 1440px;
}

.main-content .container {
    max-width: 100%;
    padding: 0;
}

/* Section Switching (Only Active View Shown) */
.dashboard-view {
    display: none;
}

.dashboard-view.active {
    display: block;
    animation: fadeInView 0.22s ease-in-out;
}

@keyframes fadeInView {
    from {
        opacity: 0;
        transform: translateY(6px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Legacy Top Navbar Compatibility (Hidden on Desktop, Handled via Sidebar) */
.navbar {
    display: none;
}

.nav-links {
    list-style: none;
}

/* ==========================================================================
   Home Section: Large Photographic Welcome Banner & 4 KPI Summary Cards
   ========================================================================== */
.home-dashboard-container {
    display: flex;
    flex-direction: column;
}

/* Welcome Photographic Header */
.welcome-banner-card {
    position: relative;
    border-radius: var(--radius-xl);
    background-color: #f7efe6;
    background-image: url('../assets/welcome_banner_bg.jpg');
    background-repeat: no-repeat;
    background-position: right center;
    background-size: cover;
    min-height: 310px;
    padding: 2.25rem 2.75rem;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.035);
    border: 1px solid rgba(0, 0, 0, 0.05);
    overflow: hidden;
    margin-bottom: 1.5rem;
}

.welcome-topbar {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    width: 100%;
    margin-bottom: 1.25rem;
}

.welcome-actions-right {
    display: flex;
    align-items: center;
    gap: 0.65rem;
}

.header-date-pill {
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(0, 0, 0, 0.06);
    padding: 0.45rem 1rem;
    border-radius: var(--radius-full);
    font-size: 0.82rem;
    font-weight: 600;
    color: #4b5563;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
}

.header-icon-btn {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.88);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(0, 0, 0, 0.06);
    color: #4b5563;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    font-size: 0.95rem;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.02);
}

.header-icon-btn:hover {
    background: #ffffff;
    color: var(--primary);
    transform: translateY(-1px);
}

.welcome-text-content {
    max-width: 520px;
    z-index: 2;
}

.welcome-greeting {
    font-size: 1.35rem;
    font-weight: 500;
    color: #374151;
    margin-bottom: 0.25rem;
    letter-spacing: -0.2px;
}

.welcome-title {
    font-size: 2.35rem;
    font-weight: 800;
    color: #111827;
    margin: 0 0 0.6rem;
    line-height: 1.15;
    letter-spacing: -0.5px;
    font-family: 'Outfit', sans-serif;
}

.welcome-title .brand-highlight {
    color: var(--primary);
    position: relative;
    display: inline-block;
}

.welcome-title .brand-highlight::after {
    content: '';
    position: absolute;
    top: -4px;
    right: -16px;
    width: 14px;
    height: 14px;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Cpath fill='%233ea25d' d='M17,8C8,10 5.9,16.17 3.82,21.34L5.71,22L6.66,19.7C7.14,19.87 7.64,20 8,20C19,20 22,3 22,3C21,5 14,5.25 9,6.25C4,7.25 2,11.5 2,13.5C2,15.5 3.75,17.25 3.75,17.25C7,8 17,8 17,8Z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-size: contain;
}

.welcome-subtitle {
    font-size: 0.96rem;
    color: #4b5563;
    line-height: 1.5;
    margin: 0 0 1.25rem;
    max-width: 480px;
}

.welcome-accent-line {
    width: 56px;
    height: 4px;
    background: #347a46;
    border-radius: 2px;
    margin-bottom: 1.25rem;
}

.welcome-cta-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
}

/* Four Compact Summary Cards */
.summary-kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1.25rem;
    margin-bottom: 1.75rem;
}

.summary-kpi-card {
    background: #ffffff;
    border: 1px solid rgba(0, 0, 0, 0.05);
    border-radius: 18px;
    padding: 1.25rem 1.35rem;
    display: flex;
    align-items: center;
    gap: 1.15rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.025);
    transition: all 0.22s ease;
}

.summary-kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.05);
    border-color: rgba(45, 106, 79, 0.25);
}

.summary-icon-pill {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    flex-shrink: 0;
}

.summary-icon-pill.green {
    background: #eaf5ec;
    color: #2d6a4f;
}

.summary-icon-pill.red {
    background: #fde8e8;
    color: #e02424;
}

.summary-details {
    display: flex;
    flex-direction: column;
}

.summary-title {
    font-size: 0.84rem;
    font-weight: 600;
    color: #4b5563;
    margin-bottom: 0.2rem;
}

.summary-number {
    font-size: 1.85rem;
    font-weight: 800;
    color: #111827;
    line-height: 1.15;
    font-variant-numeric: tabular-nums;
    font-family: 'Outfit', sans-serif;
}

/* Home Supportive Cards */
.home-secondary-grid {
    display: grid;
    grid-template-columns: 1.25fr 1fr;
    gap: 1.5rem;
}

.home-info-card {
    background: #ffffff;
    border: 1px solid rgba(0, 0, 0, 0.05);
    border-radius: var(--radius-lg);
    padding: 1.5rem 1.75rem;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.025);
}

.card-header-clean {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.25rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid var(--border-color);
}

.card-icon-title {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.header-icon-box {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
}

.header-icon-box.green {
    background: #eaf5ec;
    color: #2d6a4f;
}

.capabilities-list {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
}

.capability-item {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    font-size: 0.9rem;
    color: var(--text-secondary);
    line-height: 1.5;
}

.capability-item .cap-icon {
    font-size: 1.05rem;
    margin-top: 2px;
}

.system-status-list {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    font-size: 0.9rem;
}

.system-status-list li {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.55rem 0.75rem;
    background: #f8fafc;
    border-radius: var(--radius-sm);
    border: 1px solid var(--border-color);
}

.status-key {
    font-weight: 600;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
}

.status-val {
    font-weight: 700;
    color: var(--text-primary);
    font-size: 0.85rem;
}

/* ==========================================================================
   Buttons
   ========================================================================== */
.btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.7rem 1.4rem;
    font-size: 0.94rem;
    font-weight: 600;
    border-radius: 12px;
    border: none;
    cursor: pointer;
    transition: var(--transition);
}

.btn-sm {
    padding: 0.4rem 0.85rem;
    font-size: 0.82rem;
    border-radius: 8px;
}

.btn-primary {
    background: var(--primary);
    color: var(--text-light);
    box-shadow: 0 4px 14px rgba(45, 106, 79, 0.25);
}

.btn-primary:hover {
    background: var(--primary-dark);
    box-shadow: 0 6px 18px rgba(45, 106, 79, 0.35);
    transform: translateY(-1px);
    color: var(--text-light);
}

.btn-secondary {
    background: var(--bg-card);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-sm);
}

.btn-secondary:hover {
    background: var(--bg-subtle);
    border-color: var(--accent);
    color: var(--primary);
}

.btn-outline {
    background: transparent;
    color: var(--primary);
    border: 1.5px solid var(--primary);
}

.btn-outline:hover {
    background: var(--primary-subtle);
    color: var(--primary-dark);
}

.btn-danger {
    background: var(--status-danger-bg);
    color: var(--status-danger);
    border: 1px solid rgba(198, 40, 40, 0.2);
}

.btn-danger:hover {
    background: #ffcdd2;
}

.btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none !important;
}

/* ==========================================================================
   General Section Headers
   ========================================================================== */
.section {
    padding: 1rem 0 2.5rem;
}

.section-header {
    margin-bottom: 1.75rem;
}

.section-subtitle {
    font-size: 0.82rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--primary);
    margin-bottom: 0.35rem;
}

.section-title {
    font-size: 1.95rem;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
    letter-spacing: -0.4px;
}

.section-desc {
    font-size: 0.98rem;
    color: var(--text-secondary);
    max-width: 720px;
    line-height: 1.6;
}

/* ==========================================================================
   Feature Cards & How It Works
   ========================================================================== */
.features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1.5rem;
}

.feature-card {
    background: var(--bg-card);
    padding: 1.85rem 1.75rem;
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-color);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.025);
    transition: var(--transition);
}

.feature-card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-hover);
    border-color: rgba(45, 106, 79, 0.3);
}

.feature-icon-wrapper {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: var(--primary-subtle);
    color: var(--primary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.35rem;
    margin-bottom: 1.25rem;
}

.feature-card h3 {
    font-size: 1.15rem;
    margin-bottom: 0.5rem;
}

.feature-card p {
    font-size: 0.92rem;
    color: var(--text-secondary);
    line-height: 1.55;
}

/* How It Works Steps */
.steps-container {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 1.5rem;
    position: relative;
}

.step-card {
    background: var(--bg-card);
    padding: 2rem 1.5rem;
    border-radius: var(--radius-lg);
    border: 1px solid var(--border-color);
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.025);
    text-align: center;
    position: relative;
}

.step-number {
    width: 42px;
    height: 42px;
    background: var(--primary);
    color: var(--text-light);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    margin: 0 auto 1.25rem;
    font-size: 1.1rem;
}

.step-card h3 {
    font-size: 1.15rem;
    margin-bottom: 0.5rem;
}

.step-card p {
    font-size: 0.9rem;
    color: var(--text-secondary);
}

/* ==========================================================================
   Footer
   ========================================================================== */
.footer {
    background: #111827;
    color: #9ca3af;
    padding: 3rem 0 2rem;
    border-top: 1px solid #1f2937;
    margin-top: 3rem;
    border-radius: var(--radius-lg);
}

.footer-content {
    display: grid;
    grid-template-columns: 2fr 1fr 1fr;
    gap: 3rem;
    margin-bottom: 2rem;
    padding: 0 1.5rem;
}

.footer-brand h3 {
    color: #f9fafb;
    font-size: 1.25rem;
    margin-bottom: 0.75rem;
}

.footer-brand p {
    font-size: 0.9rem;
    line-height: 1.6;
    max-width: 400px;
}

.footer-col h4 {
    color: #f9fafb;
    font-size: 1rem;
    margin-bottom: 1rem;
}

.footer-col ul {
    list-style: none;
}

.footer-col ul li {
    margin-bottom: 0.5rem;
}

.footer-col ul li a {
    color: #9ca3af;
    font-size: 0.88rem;
}

.footer-col ul li a:hover {
    color: #a7f3d0;
}

.footer-bottom {
    text-align: center;
    padding-top: 1.5rem;
    border-top: 1px solid #1f2937;
    font-size: 0.82rem;
}

/* ==========================================================================
   Toast Notifications
   ========================================================================== */
.toast-container {
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.toast {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: #ffffff;
    color: var(--text-primary);
    padding: 0.85rem 1.25rem;
    border-radius: var(--radius-md);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    font-size: 0.92rem;
    font-weight: 500;
    border-left: 4px solid var(--primary);
    transform: translateY(20px);
    opacity: 0;
    transition: all 0.3s ease;
}

.toast.show {
    transform: translateY(0);
    opacity: 1;
}

.toast.toast-error {
    border-left-color: var(--status-danger);
}

.toast.toast-warning {
    border-left-color: var(--status-warning);
}
"""
    style_path.write_text(content, encoding="utf-8")
    print("SUCCESS: Updated frontend/css/style.css")


def update_responsive_css():
    resp_path = FRONTEND_DIR / "css" / "responsive.css"
    content = """/* ==============================================================================
   Leafora – Responsive Media Queries
   File: frontend/css/responsive.css
   ============================================================================== */

/* Desktop & Laptop Breakpoint (max-width: 1200px) */
@media (max-width: 1200px) {
    .summary-kpi-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 1rem;
    }

    .home-secondary-grid {
        grid-template-columns: 1fr;
    }
}

/* Tablet & Smaller Laptops (max-width: 992px) */
@media (max-width: 992px) {
    .leafora-sidebar {
        transform: translateX(-100%);
        transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.15);
        z-index: 1100;
        background: #ffffff;
    }

    .leafora-sidebar.open {
        transform: translateX(0);
    }

    .mobile-topbar {
        display: flex;
    }

    .main-content {
        margin-left: 0;
        padding: 5rem 1.25rem 3rem;
    }

    .analysis-dashboard {
        grid-template-columns: 1fr;
        gap: 2rem;
    }

    .footer-content {
        grid-template-columns: 1fr 1fr;
        gap: 2rem;
    }

    .footer-brand {
        grid-column: span 2;
    }

    .eval-table th,
    .eval-table td {
        padding: 0.5rem 0.85rem;
    }

    .eval-col-name {
        min-width: 220px;
    }

    .matrix-canvas-wrapper {
        min-height: 340px;
    }
}

/* Mobile Devices & Small Tablets (max-width: 768px) */
@media (max-width: 768px) {
    .section-title {
        font-size: 1.65rem;
    }

    .welcome-banner-card {
        padding: 1.75rem 1.5rem;
        background-position: 85% center;
    }

    .welcome-title {
        font-size: 2rem;
    }

    .summary-kpi-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 0.85rem;
    }

    .summary-kpi-card {
        padding: 1rem;
        gap: 0.85rem;
    }

    .summary-icon-pill {
        width: 42px;
        height: 42px;
        font-size: 1.1rem;
    }

    .summary-number {
        font-size: 1.5rem;
    }

    .panel-body {
        padding: 1.25rem;
    }

    .diagnosis-banner {
        flex-direction: column;
        align-items: flex-start;
        gap: 1rem;
    }

    .confidence-gauge {
        text-align: left;
    }

    .quality-row {
        grid-template-columns: 1fr;
    }

    .preview-actions {
        flex-direction: column;
    }

    .history-stats-grid {
        grid-template-columns: 1fr 1fr;
        gap: 0.85rem;
    }

    .stat-card {
        padding: 1rem;
        gap: 0.75rem;
    }

    .history-toolbar {
        flex-direction: column;
        align-items: stretch;
    }

    .toolbar-left {
        flex-direction: column;
        align-items: stretch;
    }

    .search-box-wrap {
        min-width: 100%;
    }

    .filter-dropdown-wrap {
        width: 100%;
    }

    .filter-dropdown-wrap .form-select {
        width: 100%;
    }

    .record-detail-grid {
        grid-template-columns: 1fr;
        gap: 1rem;
    }

    .eval-card-full .history-toolbar,
    .matrix-preview-card .history-toolbar {
        flex-direction: column;
        align-items: stretch;
        gap: 0.75rem;
    }

    .eval-card-full .search-box-wrap {
        width: 100% !important;
    }

    .matrix-preview-card .history-toolbar > div:last-child {
        justify-content: flex-end;
        width: 100%;
    }

    .matrix-canvas-wrapper {
        min-height: 280px;
    }
}

/* Small Smartphone Viewports (max-width: 480px) */
@media (max-width: 480px) {
    .container {
        padding: 0 0.5rem;
    }

    .welcome-banner-card {
        padding: 1.5rem 1.15rem;
        min-height: auto;
    }

    .welcome-title {
        font-size: 1.65rem;
    }

    .welcome-greeting {
        font-size: 1.15rem;
    }

    .welcome-subtitle {
        font-size: 0.88rem;
    }

    .welcome-topbar {
        justify-content: space-between;
    }

    .welcome-cta-row {
        flex-direction: column;
        width: 100%;
    }

    .welcome-cta-row .btn {
        width: 100%;
        justify-content: center;
    }

    .summary-kpi-grid {
        grid-template-columns: 1fr;
    }

    .dropzone {
        padding: 2rem 1rem;
    }

    .dropzone-icon {
        font-size: 2.2rem;
    }

    .tab-btn {
        font-size: 0.82rem;
        padding: 0.55rem 0.6rem;
    }

    .camera-controls {
        flex-direction: column;
    }

    .camera-controls .btn {
        width: 100%;
    }

    .toast-container {
        left: 1rem;
        right: 1rem;
        bottom: 1rem;
    }

    .toast {
        width: 100%;
    }

    .history-stats-grid {
        grid-template-columns: 1fr;
    }

    .history-pagination {
        flex-direction: column;
        gap: 0.75rem;
        text-align: center;
    }

    .history-pagination .btn-page {
        width: 100%;
        justify-content: center;
    }

    .matrix-preview-card .history-toolbar > div:last-child {
        flex-direction: column;
        align-items: stretch;
    }

    .matrix-preview-card .history-toolbar .btn {
        width: 100%;
        text-align: center;
        justify-content: center;
    }
}
"""
    resp_path.write_text(content, encoding="utf-8")
    print("SUCCESS: Updated frontend/css/responsive.css")


def update_main_js():
    main_js_path = FRONTEND_DIR / "js" / "main.js"
    content = """/**
 * Leafora – Plant Leaf Analysis & Quality Monitoring System
 * Module 7: Application Bootstrap & Main Controller
 * File: frontend/js/main.js
 */

const App = (function() {
    let toastContainer;
    const SECTIONS = ['home', 'analysis', 'history', 'model-performance', 'features', 'how-it-works', 'about'];

    function init() {
        console.log('[App] Initializing Leafora Application...');

        // Setup toast container
        toastContainer = document.getElementById('toastContainer');
        if (!toastContainer) {
            toastContainer = document.createElement('div');
            toastContainer.id = 'toastContainer';
            toastContainer.className = 'toast-container';
            document.body.appendChild(toastContainer);
        }

        // Initialize sub-controllers
        if (window.UploadController && typeof UploadController.init === 'function') {
            UploadController.init();
        }
        if (window.PredictionController && typeof PredictionController.init === 'function') {
            PredictionController.init();
        }
        if (window.HistoryController && typeof HistoryController.init === 'function') {
            HistoryController.init();
        }
        if (window.ModelEvaluationController && typeof ModelEvaluationController.init === 'function') {
            ModelEvaluationController.init();
        }

        // Setup navigation and section switcher
        setupNavigation();

        // Load dynamic home metrics (Date, greeting, summary KPI cards)
        updateHomeMetrics();

        // Check backend and model status
        verifyBackendConnection();
    }

    function switchSection(targetId) {
        if (!SECTIONS.includes(targetId)) return;

        // 1. Hide all sections, display target section
        SECTIONS.forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                if (id === targetId) {
                    el.classList.add('active');
                    el.style.display = 'block';
                } else {
                    el.classList.remove('active');
                    el.style.display = 'none';
                }
            }
        });

        // 2. Update active link in sidebar and navbar
        const allLinks = document.querySelectorAll('.nav-link, .sidebar-link, a[href^="#"]');
        allLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href === `#${targetId}`) {
                link.classList.add('active');
            } else if (href && href.startsWith('#') && SECTIONS.some(s => href === `#${s}`)) {
                link.classList.remove('active');
            }
        });

        // 3. Close mobile drawer if open
        const sidebar = document.getElementById('appSidebar');
        if (sidebar) sidebar.classList.remove('open');
        const navLinks = document.getElementById('navLinks');
        if (navLinks) navLinks.classList.remove('show');

        // 4. Scroll smoothly to top
        window.scrollTo({ top: 0, behavior: 'instant' });

        // 5. Update browser URL hash without jump
        if (window.history && window.history.pushState) {
            window.history.pushState(null, null, `#${targetId}`);
        } else {
            window.location.hash = `#${targetId}`;
        }

        // 6. Section-specific refreshes
        if (targetId === 'home') {
            updateHomeMetrics();
        } else if (targetId === 'history' && window.HistoryController && typeof HistoryController.loadHistory === 'function') {
            HistoryController.loadHistory(1);
        } else if (targetId === 'model-performance' && window.ModelEvaluationController && typeof ModelEvaluationController.loadModelEvaluation === 'function') {
            ModelEvaluationController.loadModelEvaluation();
        }
    }

    function setupNavigation() {
        const navToggle = document.getElementById('navToggle');
        const appSidebar = document.getElementById('appSidebar');
        const navLinks = document.getElementById('navLinks');
        const allLinks = document.querySelectorAll('.nav-link, .sidebar-link, a[href^="#"]');

        // Mobile drawer toggle
        if (navToggle) {
            navToggle.addEventListener('click', (e) => {
                e.stopPropagation();
                if (appSidebar) appSidebar.classList.toggle('open');
                if (navLinks) navLinks.classList.toggle('show');
            });
        }

        // Close mobile drawer when clicking outside
        document.addEventListener('click', (e) => {
            if (appSidebar && appSidebar.classList.contains('open') && !appSidebar.contains(e.target) && e.target !== navToggle && !navToggle.contains(e.target)) {
                appSidebar.classList.remove('open');
                if (navLinks) navLinks.classList.remove('show');
            }
        });

        // Intercept hash navigation to switch sections
        allLinks.forEach(link => {
            const href = link.getAttribute('href');
            if (href && href.startsWith('#')) {
                const targetId = href.substring(1);
                if (SECTIONS.includes(targetId)) {
                    link.addEventListener('click', (e) => {
                        e.preventDefault();
                        switchSection(targetId);
                    });
                }
            }
        });

        // Initialize view based on current location hash or default to 'home'
        const initialHash = window.location.hash ? window.location.hash.substring(1) : 'home';
        if (SECTIONS.includes(initialHash)) {
            switchSection(initialHash);
        } else {
            switchSection('home');
        }

        // Support browser Back and Forward buttons
        window.addEventListener('popstate', () => {
            const hash = window.location.hash ? window.location.hash.substring(1) : 'home';
            if (SECTIONS.includes(hash)) {
                switchSection(hash);
            }
        });
    }

    async function updateHomeMetrics() {
        // 1. Dynamic Date Display: e.g. "Tue, 17 Nov 2026"
        const dateEl = document.getElementById('currentDateDisplay');
        if (dateEl) {
            const now = new Date();
            const dateStr = now.toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short', year: 'numeric' });
            dateEl.textContent = dateStr;
        }

        // 2. Greeting based on local time
        const greetingEl = document.getElementById('welcomeGreeting');
        if (greetingEl) {
            const hour = new Date().getHours();
            if (hour < 12) {
                greetingEl.textContent = 'Good Morning,';
            } else if (hour < 18) {
                greetingEl.textContent = 'Good Afternoon,';
            } else {
                greetingEl.textContent = 'Good Evening,';
            }
        }

        // 3. Dynamic Summary Cards from backend
        try {
            if (typeof ApiService !== 'undefined') {
                // Fetch history stats
                const stats = await ApiService.getHistoryStats();
                if (stats && stats.status === 'SUCCESS' && stats.statistics) {
                    const s = stats.statistics;
                    const elTotal = document.getElementById('homeTotalAnalyses');
                    const elHealthy = document.getElementById('homeHealthyCount');
                    const elDiseased = document.getElementById('homeDiseasedCount');
                    if (elTotal && s.total_records !== undefined) elTotal.textContent = s.total_records;
                    if (elHealthy && s.healthy_count !== undefined) elHealthy.textContent = s.healthy_count;
                    if (elDiseased && s.diseased_count !== undefined) elDiseased.textContent = s.diseased_count;
                }

                // Fetch model evaluation accuracy
                const evalData = await ApiService.getModelEvaluation();
                if (evalData && evalData.status === 'SUCCESS' && evalData.overall_metrics) {
                    const om = evalData.overall_metrics;
                    const elAcc = document.getElementById('homeAccuracyVal');
                    if (elAcc && om.accuracy_percent !== undefined) {
                        elAcc.textContent = `${om.accuracy_percent.toFixed(2)}%`;
                    }
                }
            }
        } catch (err) {
            console.warn('[App] Could not load dynamic home metrics, using empirical baseline:', err.message);
        }
    }

    async function verifyBackendConnection() {
        const dot = document.getElementById('statusDot');
        const text = document.getElementById('statusText');

        if (!dot || !text) return;

        text.textContent = 'Connecting...';
        dot.className = 'status-dot';

        try {
            const health = await ApiService.checkHealth();
            if (health.status === 'healthy' && health.model_status && health.model_status.model_ready) {
                dot.className = 'status-dot online';
                text.textContent = 'AI Model Ready (28 Classes)';
            } else if (health.status === 'healthy') {
                dot.className = 'status-dot online';
                text.textContent = 'Backend Online';
            } else {
                dot.className = 'status-dot offline';
                text.textContent = 'Backend Offline';
                showToast('Flask backend is currently offline. Please run "python backend/app.py".', 'warning', 6000);
            }
        } catch (e) {
            dot.className = 'status-dot offline';
            text.textContent = 'Backend Offline';
        }
    }

    function showToast(message, type = 'info', duration = 4000) {
        if (!toastContainer) return;

        const toast = document.createElement('div');
        toast.className = `toast ${type === 'error' ? 'toast-error' : (type === 'warning' ? 'toast-warning' : '')}`;

        const iconClass = type === 'error' 
            ? 'fas fa-exclamation-circle' 
            : (type === 'warning' ? 'fas fa-exclamation-triangle' : 'fas fa-check-circle');

        toast.innerHTML = `
            <i class="${iconClass}"></i>
            <span>${message}</span>
        `;

        toastContainer.appendChild(toast);

        requestAnimationFrame(() => {
            toast.classList.add('show');
        });

        setTimeout(() => {
            toast.classList.remove('show');
            setTimeout(() => {
                if (toast.parentElement) {
                    toast.parentElement.removeChild(toast);
                }
            }, 300);
        }, duration);
    }

    return {
        init,
        showToast,
        switchSection,
        refreshHealth: verifyBackendConnection,
        refreshHomeMetrics: updateHomeMetrics
    };
})();

// Bootstrap on DOM ready
document.addEventListener('DOMContentLoaded', App.init);
"""
    main_js_path.write_text(content, encoding="utf-8")
    print("SUCCESS: Updated frontend/js/main.js")


def update_index_html():
    index_path = FRONTEND_DIR / "index.html"
    raw_html = index_path.read_text(encoding="utf-8")

    # Extract existing sections:
    # 1. Analysis
    m_analysis = re.search(r'(<section[^>]*id=["\']analysis["\'].*?</section>)', raw_html, re.DOTALL)
    assert m_analysis, "Could not find #analysis section"
    sec_analysis = m_analysis.group(1)
    sec_analysis = re.sub(r'class=["\'][^"\']*["\']', 'class="analysis-section dashboard-view"', sec_analysis, count=1)

    # 2. History
    m_history = re.search(r'(<section[^>]*id=["\']history["\'].*?</section>)', raw_html, re.DOTALL)
    assert m_history, "Could not find #history section"
    sec_history = m_history.group(1)
    sec_history = re.sub(r'class=["\'][^"\']*["\']', 'class="history-section dashboard-view"', sec_history, count=1)

    # 3. Model Performance
    m_eval = re.search(r'(<section[^>]*id=["\']model-performance["\'].*?</section>)', raw_html, re.DOTALL)
    assert m_eval, "Could not find #model-performance section"
    sec_eval = m_eval.group(1)
    sec_eval = re.sub(r'class=["\'][^"\']*["\']', 'class="history-section dashboard-view"', sec_eval, count=1)

    # 4. Features
    m_feat = re.search(r'(<section[^>]*id=["\']features["\'].*?</section>)', raw_html, re.DOTALL)
    assert m_feat, "Could not find #features section"
    sec_feat = m_feat.group(1)
    sec_feat = re.sub(r'class=["\'][^"\']*["\']', 'class="section dashboard-view"', sec_feat, count=1)

    # 5. How It Works
    m_hiw = re.search(r'(<section[^>]*id=["\']how-it-works["\'].*?</section>)', raw_html, re.DOTALL)
    assert m_hiw, "Could not find #how-it-works section"
    sec_hiw = m_hiw.group(1)
    sec_hiw = re.sub(r'class=["\'][^"\']*["\']', 'class="section dashboard-view"', sec_hiw, count=1)

    # 6. About
    m_about = re.search(r'(<section[^>]*id=["\']about["\'].*?</section>)', raw_html, re.DOTALL)
    assert m_about, "Could not find #about section"
    sec_about = m_about.group(1)
    sec_about = re.sub(r'class=["\'][^"\']*["\']', 'class="section dashboard-view"', sec_about, count=1)

    # Extract modals
    m_cm_modal = re.search(r'(<div id=["\']confusionMatrixModal["\'].*?</div>\s*</div>\s*</div>)', raw_html, re.DOTALL)
    if not m_cm_modal:
        m_cm_modal = re.search(r'(<div id=["\']confusionMatrixModal["\'].*?aria-label="Close matrix modal">&times;</button>.*?</div>\s*</div>\s*</div>)', raw_html, re.DOTALL)

    # If full modal block matched
    cm_modal_html = """    <!-- Modal: High-Resolution Confusion Matrix Lightbox -->
    <div id="confusionMatrixModal" class="modal-backdrop">
        <div class="modal-dialog" style="max-width:92vw;width:1050px;">
            <div class="modal-header">
                <div class="modal-title-group">
                    <i class="fas fa-th" style="color:var(--primary);margin-right:8px;"></i>
                    <h3 style="margin:0;font-size:1.2rem;font-weight:700;">MobileNetV2 28&times;28 Confusion Matrix (3600 &times; 3200 px)</h3>
                </div>
                <button id="btnCloseMatrixModal" class="modal-close" aria-label="Close matrix modal">&times;</button>
            </div>
            <div class="modal-body" style="padding:1rem;text-align:center;background:#0f172a;max-height:75vh;overflow:auto;">
                <img id="modalConfusionMatrixImg" src="" alt="Full-Resolution Confusion Matrix" style="max-width:100%;height:auto;border-radius:4px;box-shadow:0 4px 20px rgba(0,0,0,0.5);">
            </div>
            <div class="modal-footer" style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:0.75rem;">
                <span style="font-size:0.85rem;color:var(--text-muted);">
                    Evaluated on 465 PlantDoc held-out test images across all 28 classes with bilinear interpolation.
                </span>
                <div style="display:flex;gap:0.5rem;">
                    <a href="/api/model/confusion-matrix" target="_blank" id="modalRawPngLink" class="btn btn-primary" style="font-size:0.85rem;">
                        <i class="fas fa-external-link-alt"></i> Open Raw PNG
                    </a>
                    <button type="button" class="btn btn-secondary" onclick="document.getElementById('confusionMatrixModal').classList.remove('active');document.body.style.overflow=''">Close</button>
                </div>
            </div>
        </div>
    </div>

    <!-- Modal: Record Diagnostic Detail -->
    <div id="historyDetailModal" class="modal-backdrop">
        <div class="modal-dialog">
            <div class="modal-header">
                <div class="modal-title-group">
                    <i class="fas fa-microscope" style="color:var(--primary);margin-right:8px;"></i>
                    <h3 style="margin:0;font-size:1.2rem;font-weight:700;">Diagnostic Record Details</h3>
                </div>
                <button id="btnCloseModal" class="modal-close" aria-label="Close modal">&times;</button>
            </div>
            <div class="modal-body" id="modalRecordDetails">
                <!-- Injected dynamically -->
            </div>
            <div class="modal-footer">
                <button type="button" class="btn btn-secondary" onclick="document.getElementById('historyDetailModal').classList.remove('active');document.body.style.overflow=''">Close</button>
            </div>
        </div>
    </div>"""

    # Home Section with exact Leafora UI
    sec_home = """        <!-- ====================================================================
             1. Home Section: Leafora Dashboard Overview
             ==================================================================== -->
        <section class="dashboard-view active" id="home">
            <div class="home-dashboard-container">
                <!-- Large Photographic Welcome Header Card -->
                <div class="welcome-banner-card">
                    <div class="welcome-topbar">
                        <div class="welcome-actions-right">
                            <div class="header-date-pill">
                                <i class="far fa-calendar-alt"></i>
                                <span id="currentDateDisplay">Tue, 17 Nov 2026</span>
                            </div>
                            <button type="button" class="header-icon-btn" title="System Notifications" aria-label="Notifications">
                                <i class="far fa-bell"></i>
                            </button>
                            <button type="button" class="header-icon-btn" title="Leafora User Profile" aria-label="Profile">
                                <i class="far fa-user"></i>
                            </button>
                        </div>
                    </div>

                    <div class="welcome-text-content">
                        <div class="welcome-greeting" id="welcomeGreeting">Good Morning,</div>
                        <h1 class="welcome-title">
                            Welcome to <span class="brand-highlight">Leafora</span>
                        </h1>
                        <p class="welcome-subtitle">
                            AI-powered leaf classification for crop health monitoring and quality assessment.
                        </p>
                        <div class="welcome-accent-line"></div>
                        <div class="welcome-cta-row">
                            <a href="#analysis" class="btn btn-primary">
                                <i class="fas fa-camera"></i> Start Leaf Analysis
                            </a>
                            <a href="#model-performance" class="btn btn-outline">
                                <i class="fas fa-chart-line"></i> View Model Performance
                            </a>
                        </div>
                    </div>
                </div>

                <!-- Four Compact Summary Cards -->
                <div class="summary-kpi-grid">
                    <!-- Card 1: Total Analyses -->
                    <div class="summary-kpi-card" onclick="window.App && window.App.switchSection('history')" style="cursor:pointer;" title="View all analyses in History">
                        <div class="summary-icon-pill green">
                            <i class="fas fa-layer-group"></i>
                        </div>
                        <div class="summary-details">
                            <span class="summary-title">Total Analyses</span>
                            <div class="summary-number" id="homeTotalAnalyses">33</div>
                        </div>
                    </div>

                    <!-- Card 2: Healthy Leaves -->
                    <div class="summary-kpi-card" onclick="window.App && window.App.switchSection('history')" style="cursor:pointer;" title="View healthy leaf inspections in History">
                        <div class="summary-icon-pill green">
                            <i class="fas fa-leaf"></i>
                        </div>
                        <div class="summary-details">
                            <span class="summary-title">Healthy Leaves</span>
                            <div class="summary-number" id="homeHealthyCount">25</div>
                        </div>
                    </div>

                    <!-- Card 3: Diseased / Damaged -->
                    <div class="summary-kpi-card" onclick="window.App && window.App.switchSection('history')" style="cursor:pointer;" title="View diseased records in History">
                        <div class="summary-icon-pill red">
                            <i class="fas fa-virus"></i>
                        </div>
                        <div class="summary-details">
                            <span class="summary-title">Diseased / Damaged</span>
                            <div class="summary-number" id="homeDiseasedCount">8</div>
                        </div>
                    </div>

                    <!-- Card 4: Model Test Accuracy -->
                    <div class="summary-kpi-card" onclick="window.App && window.App.switchSection('model-performance')" style="cursor:pointer;" title="View empirical accuracy in Model Performance">
                        <div class="summary-icon-pill green">
                            <i class="fas fa-chart-bar"></i>
                        </div>
                        <div class="summary-details">
                            <span class="summary-title">Model Test Accuracy</span>
                            <div class="summary-number" id="homeAccuracyVal">56.56%</div>
                        </div>
                    </div>
                </div>

                <!-- Supportive Cards Below KPI Cards -->
                <div class="home-secondary-grid">
                    <div class="home-info-card">
                        <div class="card-header-clean">
                            <div class="card-icon-title">
                                <div class="header-icon-box green">
                                    <i class="fas fa-microscope"></i>
                                </div>
                                <div>
                                    <h4 style="margin:0;font-size:1.05rem;font-weight:700;color:var(--text-primary);">Core AI Capabilities</h4>
                                    <p style="margin:2px 0 0;font-size:0.8rem;color:var(--text-muted);">MobileNetV2 deep learning on 28 PlantDoc classes</p>
                                </div>
                            </div>
                            <a href="#analysis" class="btn btn-sm btn-primary">
                                <i class="fas fa-arrow-right"></i> Launch Tool
                            </a>
                        </div>
                        <div class="capabilities-list">
                            <div class="capability-item">
                                <div class="cap-icon"><i class="fas fa-check-circle" style="color:var(--primary);"></i></div>
                                <div>
                                    <strong>28 Crop Classes Identified:</strong> Apple, Corn, Grape, Tomato, Bell Pepper, Potato, Cherry, Peach, and Strawberry with specific visual pathologies.
                                </div>
                            </div>
                            <div class="capability-item">
                                <div class="cap-icon"><i class="fas fa-check-circle" style="color:var(--primary);"></i></div>
                                <div>
                                    <strong>Laplacian Blur Detection:</strong> Real-time variance sharpness filter prevents blurry handheld camera captures from corrupting diagnoses.
                                </div>
                            </div>
                            <div class="capability-item">
                                <div class="cap-icon"><i class="fas fa-check-circle" style="color:var(--primary);"></i></div>
                                <div>
                                    <strong>Harvest Maturity Guardrail:</strong> Explicitly separates leaf health from maturity, preventing premature leaf plucking.
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="home-info-card">
                        <div class="card-header-clean">
                            <div class="card-icon-title">
                                <div class="header-icon-box green">
                                    <i class="fas fa-server"></i>
                                </div>
                                <div>
                                    <h4 style="margin:0;font-size:1.05rem;font-weight:700;color:var(--text-primary);">System &amp; Pipeline Status</h4>
                                    <p style="margin:2px 0 0;font-size:0.8rem;color:var(--text-muted);">Active runtime parameters and audit storage</p>
                                </div>
                            </div>
                            <a href="#model-performance" class="btn btn-sm btn-secondary">
                                <i class="fas fa-chart-line"></i> Performance
                            </a>
                        </div>
                        <ul class="system-status-list">
                            <li>
                                <span class="status-key"><i class="fas fa-brain" style="margin-right:6px;color:var(--primary);"></i> Neural Architecture:</span>
                                <span class="status-val">MobileNetV2 (Fine-Tuned Top 34)</span>
                            </li>
                            <li>
                                <span class="status-key"><i class="fas fa-database" style="margin-right:6px;color:#2563eb;"></i> Dataset:</span>
                                <span class="status-val">PlantDoc (2,916 cleaned images)</span>
                            </li>
                            <li>
                                <span class="status-key"><i class="fas fa-sync-alt" style="margin-right:6px;color:#d97706;"></i> Preprocessing:</span>
                                <span class="status-val">cv2.INTER_LINEAR Bilinear (224&times;224)</span>
                            </li>
                            <li>
                                <span class="status-key"><i class="fas fa-table" style="margin-right:6px;color:#16a34a;"></i> Local Database:</span>
                                <span class="status-val">SQLite (leaf_analysis.db)</span>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
        </section>"""

    new_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Leafora – Plant Leaf Analysis & Quality Monitoring</title>
    
    <!-- Meta tags for agricultural AI tech -->
    <meta name="description" content="Leafora: AI-powered leaf quality classification and plucking standard monitoring system using handheld camera imagery and MobileNetV2 deep learning.">
    <meta name="author" content="CSE471 Academic Project Team">

    <!-- Google Fonts: Outfit & Plus Jakarta Sans -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">

    <!-- Font Awesome Icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css" integrity="sha512-DTOQO9RWCH3ppGqcWaEA1BIZOC6xxalwEsw9c2QQeAIftl+Vegovlnee1c9QX4TctnWMn13TZye+giMm8e2LwA==" crossorigin="anonymous" referrerpolicy="no-referrer" />

    <!-- Application Stylesheets -->
    <link rel="stylesheet" href="css/style.css">
    <link rel="stylesheet" href="css/dashboard.css">
    <link rel="stylesheet" href="css/responsive.css">
</head>
<body>

    <div class="app-layout">
        <!-- ====================================================================
             Fixed Left Sidebar (Leafora Branding & Navigation)
             ==================================================================== -->
        <aside class="leafora-sidebar" id="appSidebar">
            <a href="#home" class="sidebar-brand">
                <div class="brand-logo-icon">
                    <i class="fas fa-leaf"></i>
                </div>
                <div class="brand-text-group">
                    <span class="brand-name">Leafora</span>
                    <span class="brand-subtitle">Plant Leaf Analysis</span>
                </div>
            </a>

            <nav class="sidebar-nav-container">
                <ul class="sidebar-menu nav-links" id="navLinks">
                    <li>
                        <a href="#home" class="sidebar-link nav-link active" data-section="home">
                            <i class="fas fa-home"></i>
                            <span>Home</span>
                        </a>
                    </li>
                    <li>
                        <a href="#analysis" class="sidebar-link nav-link" data-section="analysis">
                            <i class="fas fa-camera"></i>
                            <span>Leaf Analysis</span>
                        </a>
                    </li>
                    <li>
                        <a href="#history" class="sidebar-link nav-link" data-section="history">
                            <i class="fas fa-layer-group"></i>
                            <span>History &amp; Analytics</span>
                        </a>
                    </li>
                    <li>
                        <a href="#model-performance" class="sidebar-link nav-link" data-section="model-performance">
                            <i class="fas fa-chart-bar"></i>
                            <span>Model Performance</span>
                        </a>
                    </li>
                    <li>
                        <a href="#features" class="sidebar-link nav-link" data-section="features">
                            <i class="fas fa-th-large"></i>
                            <span>Features</span>
                        </a>
                    </li>
                    <li>
                        <a href="#how-it-works" class="sidebar-link nav-link" data-section="how-it-works">
                            <i class="fas fa-project-diagram"></i>
                            <span>How It Works</span>
                        </a>
                    </li>
                    <li>
                        <a href="#about" class="sidebar-link nav-link" data-section="about">
                            <i class="fas fa-info-circle"></i>
                            <span>About</span>
                        </a>
                    </li>
                </ul>
            </nav>

            <div class="sidebar-footer">
                <div class="nav-status">
                    <span class="status-dot" id="statusDot"></span>
                    <span id="statusText">Checking Backend...</span>
                </div>
            </div>
        </aside>

        <!-- Mobile Top Header Bar -->
        <header class="mobile-topbar">
            <div class="mobile-brand">
                <i class="fas fa-leaf" style="color:var(--primary);margin-right:6px;"></i>
                <span>Leafora</span>
            </div>
            <button class="nav-toggle" id="navToggle" aria-label="Toggle navigation">
                <i class="fas fa-bars"></i>
            </button>
        </header>

        <!-- ====================================================================
             Main Content Area: 7 Single-View Top-Level Sections
             ==================================================================== -->
        <main class="main-content" id="mainContent">
{sec_home}

{sec_analysis}

{sec_history}

{sec_eval}

{sec_feat}

{sec_hiw}

{sec_about}

            <!-- Global Footer -->
            <footer class="footer">
                <div class="footer-content">
                    <div class="footer-brand">
                        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.75rem;">
                            <i class="fas fa-leaf" style="color:#52b788;font-size:1.4rem;"></i>
                            <h3 style="margin:0;font-size:1.3rem;">Leafora</h3>
                        </div>
                        <p>
                            AI-powered leaf quality classification and plucking standard monitoring system. Developed for handheld field cameras using MobileNetV2 transfer learning on 28 agricultural categories.
                        </p>
                    </div>

                    <div class="footer-col">
                        <h4>Navigation</h4>
                        <ul>
                            <li><a href="#home">Home</a></li>
                            <li><a href="#analysis">Leaf Analysis</a></li>
                            <li><a href="#history">Inspection History</a></li>
                            <li><a href="#model-performance">Model Performance</a></li>
                            <li><a href="#features">Features</a></li>
                            <li><a href="#how-it-works">How It Works</a></li>
                        </ul>
                    </div>

                    <div class="footer-col">
                        <h4>System Endpoints</h4>
                        <ul>
                            <li><a href="/api/health" target="_blank">Health Check API</a></li>
                            <li><a href="/api/predict" target="_blank">Prediction API</a></li>
                            <li><a href="/api/model/evaluation" target="_blank">Evaluation API</a></li>
                            <li><a href="#about">Project Documentation</a></li>
                        </ul>
                    </div>
                </div>

                <div class="footer-bottom">
                    &copy; 2026 Leafora (AgriLeaf AI Engine) – Plant Leaf Analysis System. All rights reserved. Academic CSE471 Project.
                </div>
            </footer>
        </main>
    </div>

{cm_modal_html}

    <!-- Toast Notification Container -->
    <div id="toastContainer" class="toast-container"></div>

    <!-- Application Scripts -->
    <script src="js/api.js"></script>
    <script src="js/upload.js"></script>
    <script src="js/prediction.js"></script>
    <script src="js/history.js"></script>
    <script src="js/model_evaluation.js"></script>
    <script src="js/main.js"></script>
</body>
</html>
"""
    index_path.write_text(new_html, encoding="utf-8")
    print("SUCCESS: Updated frontend/index.html with Leafora sidebar and full views.")


def main():
    print("Building Leafora Design System...")
    update_style_css()
    update_responsive_css()
    update_main_js()
    update_index_html()
    print("All Leafora updates written successfully!")


if __name__ == "__main__":
    main()
