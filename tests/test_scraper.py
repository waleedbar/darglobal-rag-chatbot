"""Offline tests for the scrapers (no network): parsing, allowlisting and PII scrubbing."""
import json

from scraper import darglobal, wasalt
from scraper.clean import scrub, visible_text

WASALT_ITEM = {
    "id": 5893390,
    "propertyInfo": {"slug": "apartment-120-sqm-with-2-bedrooms-5893390", "title": "Apartment 120 SQM with 2 Bedrooms",
                     "propertySubType": "Apartment", "propertyMainType": "Residential", "propertyFor": "sale",
                     "city": "Riyadh", "zone": "As-Saadah", "territory": "East Riyadh", "currencyType": "SAR",
                     "salePrice": "704000", "streetName": "King Fahd Rd 12", "country_code": "+966"},
    "attributes": [{"key": "noOfBedrooms", "name": "Bedrooms", "value": 2},
                   {"key": "builtUpArea", "name": "Built-Up Area", "value": "120", "unit": "SQM"}],
    "featureAmenities": [{"name": "Elevator"}, {"name": "Parking"}],
    "isVerified": True,
    "publishedAt": "2026-10-01T10:00:00Z",
    # personal / sensitive fields that must never be stored:
    "leadContactInfo": {"name": "Ahmed Agent", "phone": "+966 55 123 4567"},
    "propertyOwner": {"name": "Owner Name", "email": "owner@example.com"},
    "contactDetails": {"phone": "0551234567"},
    "location": {"lat": 24.71, "lon": 46.67},
}


def test_wasalt_listing_keeps_facts_and_drops_personal_data():
    d = wasalt.parse_listing(WASALT_ITEM, "sale")
    assert d["url"] == "https://wasalt.sa/en/property/sale/apartment-120-sqm-with-2-bedrooms-5893390"
    assert "704,000 SAR" in d["text"] and "Bedrooms: 2" in d["text"] and "Elevator" in d["text"]
    assert "As-Saadah, East Riyadh, Riyadh" in d["text"]
    blob = json.dumps(d, ensure_ascii=False)
    for secret in ["Ahmed Agent", "123 4567", "owner@example.com", "Owner Name", "0551234567", "24.71", "King Fahd"]:
        assert secret not in blob, secret


def test_wasalt_category_page_parsing():
    nd = {"props": {"pageProps": {"searchResult": {"properties": [WASALT_ITEM]},
                                  "FAQSchema": {"mainEntity": [{"name": "Average price?",
                                                                "acceptedAnswer": {"text": "About 1M SAR."}}]}}}}
    html = f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(nd)}</script></html>'

    class FakeClient:
        def get(self, url):
            class R:
                status_code, text = 200, html
            return R()
    docs = wasalt.scrape_category(FakeClient(), "/en/apartments-for-sale-in-riyadh", "sale", "Apartments in Riyadh")
    assert {d["type"] for d in docs} == {"listing", "page"}
    assert docs[0]["meta"]["category_page"].endswith("/en/apartments-for-sale-in-riyadh")


def test_scrub_removes_phones_and_emails_but_keeps_prices():
    s = scrub("Call +971 4 123 4567 or sales@darglobal.co.uk. Prices from AED 2,500,000, handover 2027, 448 sqm.")
    assert "123 4567" not in s and "@" not in s
    assert "AED 2,500,000" in s and "2027" in s and "448" in s


def test_visible_text_drops_navigation_and_boilerplate():
    html = """<html><head><title>T</title><meta name="description" content="Luxury villas"></head><body>
      <nav><a>Home</a><a>Projects</a></nav><main><h1>Neptune</h1><p>Five-bedroom villas in Riyadh.</p>
      <p>We use cookies to improve your experience.</p><ul><li>Handover December 2027</li></ul></main>
      <footer>All rights reserved</footer></body></html>"""
    title, desc, text = visible_text(html)
    assert title == "Neptune" and desc == "Luxury villas"
    assert "Five-bedroom villas" in text and "Handover December 2027" in text
    assert "cookies" not in text and "Home" not in text and "rights reserved" not in text


def test_darglobal_facts():
    f = darglobal.facts("Starting from AED 3.5M. 3 to 5 bedroom villas. Handover Q4 2027.")
    assert f["prices_published"] and "AED 3.5M" in f["prices_mentioned"][0]
    assert f["bedrooms"] == [3, 5] and f["handover"] == "Q4 2027"
    assert darglobal.facts("Contact our sales team for pricing.")["prices_published"] is False


def test_seed_list_has_all_catalogue_projects():
    class NoNet:
        def get(self, url):
            raise RuntimeError("offline")
    found = darglobal.discover_projects(NoNet())
    assert len(found) >= 38 and "https://darglobal.co.uk/neptune" in found
    assert "https://darglobal.co.uk/Sunrise-Haven-Luxury-Villas" in found  # case preserved


def test_embedded_text_fallback_for_client_rendered_pages():
    nd = {"props": {"pageProps": {"project": {"title": "Trump Tower Jeddah overlooking the Red Sea",
                                              "image": "https://cdn/x.jpg", "id": "abc123",
                                              "specs": [{"label": "Handover expected in Q4 2029"}]}}}}
    html = f'<html><body><div id="__next"></div><script id="__NEXT_DATA__" type="application/json">{json.dumps(nd)}</script></body></html>'
    t = darglobal.embedded_text(html)
    assert "Red Sea" in t and "Q4 2029" in t and "cdn" not in t and "abc123" not in t
