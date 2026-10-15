# Pickleball Court Time

A static Open Play calendar for CityPickle Times Square, Gotham Long Island City, PB Club Jersey City, Pickleball Kingdom Watchung, and Game On Plainfield.

Live at https://zhaozhichen.github.io/pickleball-court-time/

## Refresh and privacy

The page loads current data when opened or when dates change. Click **刷新空档** to request current availability again. There is no recurring polling, background job, or scheduled GitHub Action. The displayed check time comes with each source result. Source failures remain distinct from a full session or no scheduled events.

The browser reads the venues' publicly accessible booking interfaces with CORS. PodPlay uses the same anonymous Firebase visitor flow as its own booking pages; its published Firebase web configuration is in `src/public-config.json`. Those public application identifiers are not user credentials. Anonymous visitor tokens remain only in page memory. No passwords, existing booking-account tokens, attendee lists, carts, payment data, or reservation actions are used. Browsing preferences alone are saved to localStorage.

Data is for ordinary visitors; membership perks, rating qualifications, and final availability are determined by the venue. External APIs may change. If a provider discontinues browser access, this UI reports the failure instead of substituting demonstration data.

## Build

Node.js 22+:

```sh
npm install
npm run typecheck
npm run build
```

`assets/` is checked in so GitHub Pages can serve this project directly. No server runtime or environment variables are required. Serve the repository root with any static HTTP server for local development. Source is in `src/`; build with esbuild after edits. The homepage's fourth project links to this directory.

## Sources

- https://citypickle.podplay.app/community/events
- https://gotham.podplay.app/community/events
- https://pbclub.podplay.app/community/events
- https://pickleballkingdom.podplay.app/community/events
- https://ecom.roller.app/gameonnewjersey/onlinecheckout/en-us/home
