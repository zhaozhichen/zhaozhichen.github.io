import { VENUES, type Availability } from './courts';
import { getPodPlayAvailability } from './podplay';
import { getGameOnAvailability } from './gameon';

/** Browser-only read flow. Individual sources failing never become "no spaces". */
export async function getAvailability(date:string,refresh=false,signal?:AbortSignal):Promise<Availability[]> {
 if(signal?.aborted)throw new DOMException('Cancelled','AbortError');
 const results=await Promise.all(VENUES.map(async venue=>{
  try{return venue.id==='gameon'?await getGameOnAvailability(date):await getPodPlayAvailability(venue,date,refresh);}
  catch{return {venueId:venue.id,date,status:'unavailable' as const,checkedAt:new Date().toISOString(),slots:[],message:'暂时无法读取官方场次，请重试或前往官方查看。'};}
 }));
 if(signal?.aborted)throw new DOMException('Cancelled','AbortError');
 return results;
}
