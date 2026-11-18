// Run in the browser console on a public Wasalt category page, e.g.
//   https://wasalt.sa/en/apartments-for-sale-in-riyadh
// Visit pages slowly (one every ~10 s). Only allowlisted listing facts are kept, extracted inside the
// page: no owner/agent/contact fields, coordinates, street names, images or reference numbers.
// When done, run exportSnapshot() to download wasalt_snapshot.jsonl, then:
//   python -m scraper.wasalt_import import wasalt_snapshot.jsonl
function collect(purpose, label) {
  const el = document.getElementById('__NEXT_DATA__'); if (!el) return -1;
  const pp = JSON.parse(el.textContent).props.pageProps;
  const INFO = ['slug','title','propertySubType','propertyMainType','propertyFor','city','state','territory','zone',
                'currencyType','salePrice','expectedRent','unitType'];
  const nm = v => (v && typeof v === 'object') ? (v.name || v.title || v.value || '') : v;
  const props = ((pp.searchResult || {}).properties || []).map(p => {
    const info = {};
    INFO.forEach(k => { const v = nm((p.propertyInfo || {})[k]); if (v !== undefined && v !== null && v !== '') info[k] = v; });
    return { id: p.id, propertyInfo: info,
      attributes: (p.attributes || []).filter(a => a && a.name && a.value !== null && a.value !== '')
                                      .map(a => ({name: a.name, value: a.value, unit: a.unit || ''})),
      featureAmenities: (p.featureAmenities || []).map(x => ({name: nm(x)})).filter(x => x.name).slice(0, 25),
      isVerified: !!p.isVerified, isRegaProp: !!p.isRegaProp, publishedAt: p.publishedAt || null };
  });
  const faq = pp.FAQSchema && pp.FAQSchema.mainEntity
    ? pp.FAQSchema.mainEntity.map(q => ({name: q.name, acceptedAnswer: {text: (q.acceptedAnswer || {}).text}})) : [];
  const store = JSON.parse(localStorage.getItem('dg_rag_snapshot') || '{}');
  if (props.length) store[location.pathname] = { purpose, label, url: location.origin + location.pathname,
                                                 collected_at: new Date().toISOString(), properties: props, faq };
  localStorage.setItem('dg_rag_snapshot', JSON.stringify(store));
  return props.length;
}

function exportSnapshot() {
  const s = JSON.parse(localStorage.getItem('dg_rag_snapshot') || '{}');
  const blob = new Blob([Object.values(s).map(x => JSON.stringify(x)).join('\n') + '\n'], {type: 'application/x-ndjson'});
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'wasalt_snapshot.jsonl'; a.click();
}
