import { VENUES } from './courts';

export const PREFERENCES_VERSION = 2;

export function restoreSelectedVenues(prefs: { selected?: unknown; version?: number }): string[] {
  if (!Array.isArray(prefs.selected)) return VENUES.map(venue => venue.id);
  const selected = prefs.selected.filter((id): id is string => VENUES.some(venue => venue.id === id));
  // Existing Gotham followers should see its newly added branch on first load.
  // Save the new version so subsequently deselecting that branch stays respected.
  if ((prefs.version ?? 1) < PREFERENCES_VERSION && selected.includes('gotham')) selected.push('gotham-hunters');
  return [...new Set(selected)];
}
