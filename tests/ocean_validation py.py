# %% [markdown]
# # CogniNPC — Test 2: Walidacja spojnosci profilu psychologicznego (OCEAN)
#
# **Projekt eksperymentu:**
# - Ablacja C0/C1/C2: brak OCEAN / surowe liczby / trojstopniowe mapowanie
#   behawioralne (system docelowy)
# - Kazda cecha testowana osobno (pozostale 4 na wartosci neutralnej 0.5),
#   przy dwoch biegunach: wysoki (0.85) i niski (0.15)
# - Jedna neutralna postac-nosnik (Aldric), zeby wyeliminowac konfundacje
#   z backstory/quirkami istniejacych NPC
# - Sedzia (Claude via API) wykonuje DWA zadania w jednym wywolaniu:
#   1) zgadniecie bieguna (HIGH/LOW) bez znajomosci profilu -> binomtest
#      (test odzyskiwalnosci, tylko na C2 — systemie docelowym)
#   2) ciagla ocena natezenia cechy 0-100 -> adherence_score -> Mann-Whitney
#      (ablacja: czy C2 daje silniejsza, lepiej ukierunkowana ekspresje
#      cechy niz C1?)

# %%
import json
import os
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

OUT = Path("results_test2")
OUT.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. Konfiguracja eksperymentu

# %%
API_URL = "http://localhost:8000/api/chat"
NPC_ID = "aldric_neutral"  # patrz npc_profile_aldric_neutral.json

TRAITS = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
POLES = {"high": 0.85, "low": 0.15}
NEUTRAL = 0.5

CONDITIONS = ["C0_none", "C1_raw", "C2_full"]
OCEAN_MODE_MAP = {"C0_none": "none", "C1_raw": "raw", "C2_full": "full"}

REPETITIONS = 3       # smoke test: ustaw 1
COOLDOWN_S = 0.2
TIMEOUT_S = 120.0
MAX_CONCURRENT = 4     # wspolbieznosc generacji — backend juz to obsluguje

MODEL = "llama3.1:latest"   # ZWERYFIKUJ nazwe modelu (ollama list)
NUM_PREDICT = 100           # kontrolowana dlugosc — nieistotne dla oceny
                             # sedziego, ale przyspiesza przebieg

RUN_ID = uuid.uuid4().hex[:12]
RAW_PATH = OUT / f"generations_{RUN_ID}.jsonl"
print(f"run_id = {RUN_ID}")

# %% [markdown]
# ## 2. Taksonomia bodzcow "stresowych"
#
# Piec kategorii sytuacji, w ktorych rozbieznosc miedzy cechami OCEAN powinna
# byc najbardziej widoczna w tonie i tresci odpowiedzi. Losowa rotacja
# eliminuje efekt pozycji i wplyw pojedynczego, szczegolnie latwego pytania.

# %%
STIMULI: list[tuple[str, str]] = [
    ("oskarzenie", "Someone in the village claims you stole coin from the merchant guild yesterday. What do you say?"),
    ("oskarzenie", "The guard captain accuses you of lying about where you were last night. How do you respond?"),
    ("prosba", "I have no money left, but I desperately need your help crossing the mountains. Will you help me anyway?"),
    ("prosba", "My child is sick and I cannot afford a healer. Could you lend me what you have?"),
    ("niepowodzenie", "The item you gave me broke after a single use. What happened here?"),
    ("niepowodzenie", "We followed your advice and it went badly wrong. People got hurt because of it."),
    ("nowina", "I just heard the king has declared war on the northern kingdom."),
    ("nowina", "Word just arrived that the old bridge to the capital has collapsed."),
    ("presja_czasu", "Bandits are approaching and will be here in minutes — do we fight or flee? Decide now."),
    ("presja_czasu", "The tide is rising fast and the cave exit will flood soon. What do we do?"),
]

# %% [markdown]
# ## 3. Wywolanie backendu — generacja pojedynczej odpowiedzi

# %%
def build_ocean_vector(trait: str, pole: str) -> dict[str, float]:
    vec = {t: NEUTRAL for t in TRAITS}
    vec[trait] = POLES[pole]
    return vec


def generate_response(
    client: httpx.Client, trait: str, pole: str, condition: str,
    stim_category: str, stim_text: str, rep: int,
) -> dict:
    payload = {
        "npc_id": NPC_ID,
        "player_message": stim_text,
        "model": MODEL,
        "num_predict": NUM_PREDICT,
        "rag_k": 0,                    # eliminuje RAG jako zmienna zaklocajaca
        "skip_memory_write": True,
        "ocean_mode": OCEAN_MODE_MAP[condition],
        "ocean_override": build_ocean_vector(trait, pole),
    }
    t0 = time.perf_counter()
    try:
        r = client.post(API_URL, json=payload)
        wall_ms = (time.perf_counter() - t0) * 1000
        r.raise_for_status()
        body = r.json()
        return {
            "ok": True,
            "trait": trait,
            "pole": pole,
            "condition": condition,
            "stim_category": stim_category,
            "stim_text": stim_text,
            "rep": rep,
            "response_text": body.get("response_text", ""),
            "wall_ms": round(wall_ms, 1),
        }
    except Exception as exc:
        return {
            "ok": False, "trait": trait, "pole": pole, "condition": condition,
            "stim_category": stim_category, "stim_text": stim_text, "rep": rep,
            "error": f"{type(exc).__name__}: {exc}",
        }

# %% [markdown]
# ## 4. Glowna petla generacji
#
# Siatka: 5 cech x 2 bieguny x 3 warunki x 10 bodzcow x REPETITIONS.
# Przy domyslnych ustawieniach (REPETITIONS=3): 900 generacji.

# %%
grid = list(product(TRAITS, POLES.keys(), CONDITIONS, STIMULI, range(REPETITIONS)))
print(f"Zaplanowanych generacji: {len(grid)}")

t_start = time.perf_counter()
with RAW_PATH.open("w", encoding="utf-8") as fh, \
     httpx.Client(timeout=TIMEOUT_S) as client, \
     ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as ex:

    futures = []
    for trait, pole, cond, (cat, text), rep in grid:
        futures.append(ex.submit(generate_response, client, trait, pole, cond, cat, text, rep))

    done = 0
    for fut in futures:
        rec = fut.result()
        rec["run_id"] = RUN_ID
        rec["ts"] = datetime.now(timezone.utc).isoformat()
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        done += 1
        if done % 50 == 0:
            print(f"  {done}/{len(grid)}")

print(f"Zakonczono w {(time.perf_counter()-t_start)/60:.1f} min -> {RAW_PATH}")

# %% [markdown]
# ---
# ## 5. Sedzia LLM — ocena odpowiedzi
#
# Osobny, wznawialny krok: czyta surowe generacje, dopisuje osady sedziego
# do NOWEGO pliku. Mozna przerwac i wznowic bez ponownej generacji.
#
# UZYWA GEMINI (google-genai) zamiast Claude — SDK ma natywny structured
# output (response_schema), wiec nie trzeba recznie obcinac ogrodzen
# markdown jak w wersji z Claude. Metodologicznie rownowazne: generator
# (Llama 3.1) jest spoza obu rodzin sedziow, wiec nie ma ryzyka
# self-preference bias niezaleznie od wyboru.
#
# Wymaga: `pip install google-genai` oraz zmiennej srodowiskowej
# `GEMINI_API_KEY` (klucz z Google AI Studio).
#
# UWAGA NA LIMITY: subskrypcja Google AI Pro podnosi limity glownie w
# interfejsie AI Studio Playground/Build, NIE gwarantuje automatycznie
# wysokich limitow dla wywolan programistycznych z klucza API. Sprawdz
# aktualny RPM/RPD dla swojego projektu w zakladce "Rate Limits" w AI
# Studio PRZED odpaleniem pelnej siatki (900 wywolan) — ponizszy kod ma
# wbudowany backoff na bledy 429, ale przy bardzo niskim RPM (np. 5-15,
# typowe dla darmowego/podstawowego tieru modeli Pro) pelny przebieg moze
# zajac godzine lub wiecej. Rozwazenie modelu Flash zamiast Pro jako
# sedziego jest tania opcja przyspieszenia, kosztem nieco slabszej jakosci
# oceny — do przetestowania na male probce przed decyzja.

# %%
from google import genai
from google.genai import types

JUDGE_MODEL = "gemini-2.5-pro"  # ZWERYFIKUJ aktualna nazwe modelu w AI Studio
JUDGED_PATH = OUT / f"judged_{RUN_ID}.jsonl"

judge_client = genai.Client()  # czyta GEMINI_API_KEY (lub GOOGLE_API_KEY) ze srodowiska

TRAIT_DESCRIPTIONS = {
    "openness": "Openness to Experience (curiosity, imagination, unconventionality vs. conventional, practical, routine-preferring)",
    "conscientiousness": "Conscientiousness (organized, disciplined, careful vs. impulsive, careless, disorganized)",
    "extraversion": "Extraversion (outgoing, energetic, talkative vs. reserved, quiet, withdrawn)",
    "agreeableness": "Agreeableness (warm, cooperative, trusting vs. cold, suspicious, antagonistic)",
    "neuroticism": "Neuroticism (anxious, emotionally volatile, easily distressed vs. calm, emotionally stable, secure)",
}

JUDGE_SYSTEM_PROMPT = """You are a psychology researcher analyzing dialogue for personality trait markers.
You will be shown a single line of dialogue spoken by a fictional character, and told which
Big Five (OCEAN) trait dimension to focus on. You do NOT know the character's assigned
personality profile — infer purely from the text itself."""

# Schemat structured output — Gemini wymusza zgodnosc odpowiedzi z tym
# schematem, wiec json.loads() nizej nigdy nie dostanie zniekształconego
# JSON-a (w przeciwienstwie do wersji z recznym parsowaniem tekstu Claude).
JUDGE_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "estimated_level": types.Schema(type=types.Type.STRING, enum=["high", "low"]),
        "intensity_0_100": types.Schema(type=types.Type.INTEGER),
        "rationale": types.Schema(type=types.Type.STRING),
    },
    required=["estimated_level", "intensity_0_100", "rationale"],
)


def judge_response(trait: str, response_text: str) -> dict:
    user_msg = (
        f"Trait dimension to assess: {TRAIT_DESCRIPTIONS[trait]}\n\n"
        f'Dialogue line: "{response_text}"\n\n'
        "Based ONLY on this line, estimate the level of this trait. "
        "0 = trait very low/absent, 100 = trait very strongly present."
    )
    resp = judge_client.models.generate_content(
        model=JUDGE_MODEL,
        contents=user_msg,
        config=types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=JUDGE_RESPONSE_SCHEMA,
            temperature=0.0,  # deterministyczna ocena — sedzia nie powinien "kreatywnie" zgadywac
        ),
    )
    return json.loads(resp.text)


def judge_with_retry(trait: str, response_text: str, max_retries: int = 3) -> dict | None:
    for attempt in range(max_retries):
        try:
            return judge_response(trait, response_text)
        except Exception as exc:  # obejmuje bledy 429 (rate limit) i inne API errors
            is_rate_limit = "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
            if attempt == max_retries - 1:
                print(f"  [judge] blad po {max_retries} probach: {exc}")
                return None
            # Dluzszy backoff przy 429 — limity RPM resetuja sie co minute
            wait = 20.0 if is_rate_limit else 2 ** attempt
            time.sleep(wait)
    return None

# %%
# Wznawialnosc: zbierz juz osadzone (trait, response_text) hashe, pomin je.
already_judged: set[tuple] = set()
if JUDGED_PATH.exists():
    for line in JUDGED_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            already_judged.add((r["trait"], r["response_text"]))
    print(f"Wznowienie: {len(already_judged)} juz osadzonych")

generations = [json.loads(l) for l in RAW_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
generations = [g for g in generations if g.get("ok") and g.get("response_text")]
to_judge = [g for g in generations if (g["trait"], g["response_text"]) not in already_judged]
print(f"Do osadzenia: {len(to_judge)} / {len(generations)}")

with JUDGED_PATH.open("a", encoding="utf-8") as fh:
    for i, gen in enumerate(to_judge, 1):
        verdict = judge_with_retry(gen["trait"], gen["response_text"])
        if verdict is None:
            continue
        row = {**gen, **verdict}
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        if i % 25 == 0:
            print(f"  osadzono {i}/{len(to_judge)}")

print(f"Gotowe -> {JUDGED_PATH}")
# %% [markdown]
# ## 6. Analiza
#
# Czyta wylacznie plik JSONL sedziego — mozna uruchamiac wielokrotnie bez
# ponownego generowania czy oceniania.

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binomtest, mannwhitneyu

sns.set_theme(style="whitegrid", context="paper", font_scale=1.05)
plt.rcParams["figure.dpi"] = 150

df = pd.DataFrame([json.loads(l) for l in JUDGED_PATH.read_text(encoding="utf-8").splitlines() if l.strip()])
print(f"Wczytano {len(df)} osadzonych odpowiedzi")

# Adherence score: znormalizowany tak, by WYZSZA wartosc ZAWSZE oznaczala
# "bliżej przypisanego bieguna", niezaleznie czy to high czy low.
# Dzieki temu mozna porownywac warunki (C0/C1/C2) na jednej skali, laczac
# przypadki high i low w jeden rozklad per warunek.
df["adherence_score"] = np.where(
    df["pole"] == "high", df["intensity_0_100"], 100 - df["intensity_0_100"]
)
df["correct"] = df["estimated_level"] == df["pole"]

# %% [markdown]
# ### 6.1 Test odzyskiwalnosci (recoverability) — binomtest, tylko C2
#
# Pytanie: czy sedzia, widzac WYLACZNIE tekst (bez profilu), zgaduje biegun
# cechy lepiej niz rzut moneta? To bezposrednia walidacja, czy trojstopniowe
# mapowanie behawioralne faktycznie manifestuje sie w wypowiedziach.

# %%
def recoverability_report(df_in: pd.DataFrame) -> pd.DataFrame:
    rows = []
    d = df_in[df_in["condition"] == "C2_full"]
    for trait, g in d.groupby("trait"):
        hits = int(g["correct"].sum())
        n = len(g)
        res = binomtest(hits, n, p=0.5, alternative="greater")
        rows.append({
            "trait": trait, "n": n, "hits": hits,
            "accuracy": round(hits / n, 3),
            "p_value": round(res.pvalue, 5),
            "istotne_p<.05": res.pvalue < 0.05,
        })
    # agregat po wszystkich cechach lacznie
    hits_all, n_all = int(d["correct"].sum()), len(d)
    res_all = binomtest(hits_all, n_all, p=0.5, alternative="greater")
    rows.append({
        "trait": "WSZYSTKIE", "n": n_all, "hits": hits_all,
        "accuracy": round(hits_all / n_all, 3),
        "p_value": round(res_all.pvalue, 8),
        "istotne_p<.05": res_all.pvalue < 0.05,
    })
    return pd.DataFrame(rows)


recov = recoverability_report(df)
recov.to_csv(OUT / "tables_recoverability.csv", index=False)
print("=== Test odzyskiwalnosci (C2, system docelowy) ===")
print(recov.to_string(index=False))

# %% [markdown]
# ### 6.2 Ablacja — Mann-Whitney U na adherence_score
#
# Pytanie: czy pelny system (C2) daje SILNIEJSZA, lepiej ukierunkowana
# ekspresje cechy niz surowa injekcja liczb (C1)? A czy C1 w ogole daje
# cokolwiek ponad brak informacji (C0)?

# %%
def ablation_report(df_in: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for trait, g in df_in.groupby("trait"):
        c0 = g[g["condition"] == "C0_none"]["adherence_score"].to_numpy()
        c1 = g[g["condition"] == "C1_raw"]["adherence_score"].to_numpy()
        c2 = g[g["condition"] == "C2_full"]["adherence_score"].to_numpy()

        def cmp(a, b, label):
            if len(a) < 3 or len(b) < 3:
                return None
            u, p = mannwhitneyu(a, b, alternative="greater")
            return {
                "trait": trait, "porownanie": label,
                "n_a": len(a), "n_b": len(b),
                "med_a": round(float(np.median(a)), 1),
                "med_b": round(float(np.median(b)), 1),
                "U": u, "p_value": round(p, 5), "istotne_p<.05": p < 0.05,
            }

        for res in (cmp(c2, c1, "C2 > C1"), cmp(c2, c0, "C2 > C0"), cmp(c1, c0, "C1 > C0")):
            if res:
                rows.append(res)
    return pd.DataFrame(rows)


ablation = ablation_report(df)
ablation.to_csv(OUT / "tables_ablation.csv", index=False)
print("\n=== Ablacja: Mann-Whitney U (adherence_score) ===")
print(ablation.to_string(index=False))

# %% [markdown]
# ### 6.3 Wykresy

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Recoverability per trait
sns.barplot(data=recov[recov["trait"] != "WSZYSTKIE"], x="trait", y="accuracy", ax=axes[0])
axes[0].axhline(0.5, ls="--", color="crimson", label="poziom przypadku (50%)")
axes[0].set(ylim=(0, 1), xlabel="", ylabel="Trafnosc sedziego",
            title="Test odzyskiwalnosci (C2) — trafnosc wg cechy")
axes[0].legend()
axes[0].tick_params(axis="x", rotation=30)

# Ablacja: rozklad adherence_score per warunek
sns.boxplot(data=df, x="condition", y="adherence_score",
            order=["C0_none", "C1_raw", "C2_full"], showfliers=False, ax=axes[1])
axes[1].axhline(50, ls="--", color="grey", lw=1)
axes[1].set(xlabel="", ylabel="Adherence score (0-100)",
            title="Ablacja: natezenie cechy zgodne z przypisanym biegunem")

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"rys_ocean_summary.{ext}")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(9, 5))
sns.boxplot(data=df, x="trait", y="adherence_score", hue="condition",
            hue_order=["C0_none", "C1_raw", "C2_full"], showfliers=False, ax=ax)
ax.axhline(50, ls="--", color="grey", lw=1)
ax.set(xlabel="", ylabel="Adherence score (0-100)",
       title="Ablacja per cecha: C0 vs C1 vs C2")
ax.tick_params(axis="x", rotation=20)
ax.legend(title="Warunek", fontsize=8)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"rys_ocean_per_trait.{ext}")
plt.show()

# %% [markdown]
# ### 6.4 Podsumowanie do rozdzialu wynikow

# %%
n_sig_recov = (recov["istotne_p<.05"] & (recov["trait"] != "WSZYSTKIE")).sum()
n_sig_c2c1 = ((ablation["porownanie"] == "C2 > C1") & ablation["istotne_p<.05"]).sum()

print("=" * 60)
print("PODSUMOWANIE TESTU 2")
print("=" * 60)
print(f"Cech z istotna odzyskiwalnoscia (p<.05):  {n_sig_recov}/5")
print(f"Cech gdzie C2 istotnie > C1 (p<.05):        {n_sig_c2c1}/5")
print(f"Sredni adherence_score C0/C1/C2:            "
      f"{df[df.condition=='C0_none']['adherence_score'].mean():.1f} / "
      f"{df[df.condition=='C1_raw']['adherence_score'].mean():.1f} / "
      f"{df[df.condition=='C2_full']['adherence_score'].mean():.1f}")
print("=" * 60)
print(f"\nTabele -> {OUT}\nWykresy -> {OUT}")
