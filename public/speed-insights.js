// Import and initialize Vercel Speed Insights
// Using unpkg.com CDN to load the package
import { injectSpeedInsights } from 'https://unpkg.com/@vercel/speed-insights@1.3.1/dist/index.mjs';

// Initialize Speed Insights
injectSpeedInsights({
    debug: false
});
