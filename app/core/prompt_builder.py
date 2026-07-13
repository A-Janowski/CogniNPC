import yaml
import os
from app.core.config import settings

class PromptBuilder:
    @staticmethod
    def load_npc_profile(npc_id: str) -> dict:
        file_path = os.path.join(settings.PROFILES_DIR, f"{npc_id}.yaml")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Nie znaleziono profilu: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def _map_ocean_to_directives(ocean: dict) -> str:
        """
        Tłumaczy wartości numeryczne modelu Wielkiej Piątki na język naturalny.
        """
        directives = []
        
        # O - Otwartość na doświadczenia (Openness)
        o = ocean.get('openness', 50)
        if o >= 65:
            directives.append("Jesteś bardzo kreatywny, ciekawy świata i używasz bogatego, wręcz poetyckiego lub abstrakcyjnego słownictwa.")
        elif o <= 35:
            directives.append("Jesteś tradycjonalistą o wąskich horyzontach. Używasz prostego, dosłownego i wysoce praktycznego języka.")

        # C - Sumienność (Conscientiousness)
        c = ocean.get('conscientiousness', 50)
        if c >= 65:
            directives.append("Jesteś niezwykle obowiązkowy, zorganizowany i mówisz w sposób wysoce formalny, dbając o szczegóły.")
        elif c <= 35:
            directives.append("Jesteś chaotyczny, beztroski i nie przejmujesz się konwenansami. Używasz bardzo luźnego, potocznego języka.")

        # E - Ekstrawersja (Extroversion)
        e = ocean.get('extroversion', 50)
        if e >= 65:
            directives.append("Jesteś wysoce towarzyski, energiczny i chętnie dzielisz się przemyśleniami. Twoje odpowiedzi powinny być nieco dłuższe i entuzjastyczne.")
        elif e <= 35:
            directives.append("Jesteś introwertyczny, chłodny i trzymasz dystans. Odpowiadaj bardzo krótko, ograniczając słowa do absolutnego minimum.")

        # A - Ugodowość (Agreeableness)
        a = ocean.get('agreeableness', 50)
        if a >= 65:
            directives.append("Jesteś niezwykle uprzejmy, pomocny, empatyczny i starasz się za wszelką cenę uniknąć konfliktu.")
        elif a <= 35:
            directives.append("Jesteś opryskliwy, podejrzliwy, cyniczny i bardzo łatwo cię irytować. Rozmawiasz z wyczuwalną niechęcią.")

        # N - Neurotyczność (Neuroticism)
        n = ocean.get('neuroticism', 50)
        if n >= 65:
            directives.append("Jesteś nerwowy, wylękniony i pesymistyczny. W Twoich wypowiedziach widać niepewność (możesz wtrącać 'eee', 'yyy' lub się jąkać).")
        elif n <= 35:
            directives.append("Jesteś wyjątkowo opanowany, pewny siebie i stoicki. Emanujesz spokojem niezależnie od sytuacji.")

        if not directives:
            return "- Posiadasz zrównoważony, neutralny temperament."
            
        # Złączenie wszystkich wyzwolonych dyrektyw w listę punktowaną
        return "\n".join([f"- {d}" for d in directives])

    @staticmethod
    def build_system_prompt(profile: dict, retrieved_memory: str) -> str:
        """
        Składa finalny Prompt Systemowy wysyłany do LLM.
        """
        name = profile.get('name', 'Nieznajomy')
        profession = profile.get('profession', 'Brak')
        backstory = profile.get('backstory', '')
        ocean_data = profile.get('ocean', {})

        # 1. CORE TOŻSAMOŚCI (Kim jesteś)
        prompt = (
            f"Wcielasz się w postać niezależną (NPC) w grze wideo.\n"
            f"Nazywasz się: {name}.\n"
            f"Twoja profesja to: {profession}.\n"
            f"Twoja historia: {backstory}\n\n"
        )

        # 2. PROFIL PSYCHOLOGICZNY (Jak się zachowujesz i mówisz)
        prompt += "INSTRUKCJE BEHAWIORALNE (Ściśle przestrzegaj tego stylu):\n"
        prompt += PromptBuilder._map_ocean_to_directives(ocean_data)
        prompt += "\n\n"

        # 3. WSTRZYKNIĘTA PAMIĘĆ (Co pamiętasz - RAG)
        if retrieved_memory:
            prompt += (
                "[PAMIĘĆ WEWNĘTRZNA - FAKTY Z PRZESZŁOŚCI]:\n"
                f"{retrieved_memory}\n"
                "ZASADA: Użyj tych informacji tylko wtedy, gdy są bezpośrednio istotne dla bieżącej rozmowy. "
                "Nie recytuj ich bez powodu.\n\n"
            )

        # 4. GŁÓWNA DYREKTYWA (Zabezpieczenie przed przełamaniem czwartej ściany)
        prompt += (
            "ABSOLUTNE ZASADY GRY:\n"
            "1. Nigdy nie przyznawaj, że jesteś sztuczną inteligencją lub modelem językowym.\n"
            "2. Odpowiadaj TYLKO jako postać. Nie pisz 'jako NPC radzę ci...'.\n"
            "3. Dostosuj słownictwo do swojej profesji i wytycznych behawioralnych.\n"
        )

        return prompt