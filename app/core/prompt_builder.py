import yaml
import os
from app.core.config import settings

class PromptBuilder:
    @staticmethod
    def load_npc_profile(npc_id: str) -> dict:
        file_path = os.path.join(settings.PROFILES_DIR, f"{npc_id}.yaml")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"NPC Profile not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    @staticmethod
    def map_ocean_to_directives(ocean: dict) -> str:
        """
        Translates the Big Five (OCEAN) personality traits (0-100) into 
        precise English behavioral directives using a three-tier mapping.
        """
        directives = []

        # O - Openness
        o = ocean.get('openness', 50)
        if o >= 65:
            directives.append("You are highly creative, imaginative, and intellectually curious. Use a rich, descriptive, and sometimes abstract or poetic vocabulary.")
        elif o <= 35:
            directives.append("You are a pragmatic traditionalist, highly skeptical of new ideas or magic. You value only practical and tangible things. Use simple, direct, and literal language.")
        else:
            directives.append("You have a common-sense approach to the world. You are practical but open to useful ideas. Your vocabulary is standard and grounded.")

        # C - Conscientiousness
        c = ocean.get('conscientiousness', 50)
        if c >= 65:
            directives.append("You are extremely disciplined, organized, and detail-oriented. Speak in a structured, methodical, and formal manner, focusing on precision.")
        elif c <= 35:
            directives.append("You are chaotic, carefree, and disorganized. You ignore social conventions, speak in a highly casual, unstructured manner, and easily drift from one topic to another.")
        else:
            directives.append("You are reliable in your duties but flexible in life. Speak naturally and coherently without excessive formality or chaos.")

        # E - Extroversion
        e = ocean.get('extroversion', 50)
        if e >= 65:
            directives.append("You are highly social, enthusiastic, and energetic. You easily close the distance with others. Your responses should be slightly longer and expressive.")
        elif e <= 35:
            directives.append("You are an introvert with a cold demeanor. You value quietness and keep people at a distance. Respond very briefly, using only the necessary minimum of words.")
        else:
            directives.append("You are moderately social. Speak when necessary, but you are comfortable with silence. The length of your responses is balanced and natural.")

        # A - Agreeableness
        a = ocean.get('agreeableness', 50)
        if a >= 65:
            directives.append("You are incredibly polite, warm, empathetic, and patient. You strive to help the speaker and avoid conflict at all costs.")
        elif a <= 35:
            directives.append("You are rude, suspicious, cynical, and easily irritated. Speak with clear reluctance, using sharp, sarcastic, or hostile remarks.")
        else:
            directives.append("You are assertive. You maintain basic politeness but do not let others walk over you. You help only those who prove they deserve it.")

        # N - Neuroticism
        n = ocean.get('neuroticism', 50)
        if n >= 65:
            directives.append("You are highly tense, anxious, pessimistic, and easily worried. Your speech should reflect nervousness, hesitation, or frequent stuttering (e.g., use 'uhm', 'well...', 'I-I don't know').")
        elif n <= 35:
            directives.append("You possess an absolute, stoic calmness and self-confidence. Nothing can stress or frighten you. Your responses are stable, firm, and composed.")
        else:
            directives.append("Your emotional responses are stable and fit the situation. You only show worry under real, immediate danger, remaining calm in daily life.")

        return "\n".join([f"- {d}" for d in directives])

    @staticmethod
    def build_system_prompt(profile: dict, retrieved_memory: str, ocean_mode: str = "full") -> str:
        """
        Assembles the final English System Prompt sent to the LLM.
        Includes safety guards to prevent default AI assistant patterns.

        ocean_mode:
          "full" — three-level mapping (PromptBuilder.
                   map_ocean_to_directives) — CURRENT behavior,
                   default, so that the Unity client doesn't notice any changes.
          "raw"  — PromptBuilder.format_ocean_raw — raw numbers.
          "none" — section completely omitted.
        """

        name = profile.get('name', 'Stranger')
        profession = profile.get('profession', 'None')
        backstory = profile.get('backstory', '')
        quirks = profile.get('quirks', [])
        ocean_data = profile.get('ocean', {})

        # 1. CORE IDENTITY
        prompt = (
            f"You are roleplaying as a Non-Player Character (NPC) in a fantasy RPG.\n"
            f"Your name is: {name}.\n"
            f"Your profession is: {profession}.\n"
            f"Your backstory and lore:\n{backstory}\n\n"
        )

        # 1b. UNIQUE QUIRKS
        if quirks:
            prompt += "PERSONAL QUIRKS AND SPEECH PATTERNS:\n"
            prompt += "\n".join([f"- {q}" for q in quirks]) + "\n\n"

        # 2. BEHAVIORAL DIRECTIVES (OCEAN)
        if ocean_mode == "full":
            prompt += "BEHAVIORAL STYLE (Strictly adhere to these traits):\n"
            prompt += PromptBuilder.map_ocean_to_directives(ocean_data)
            prompt += "\n\n"
        elif ocean_mode == "raw":
            prompt += PromptBuilder.format_ocean_raw(ocean_data)
            prompt += "\n\n"
        elif ocean_mode == "none":
            pass  # C0: brak jakiejkolwiek informacji o osobowosci
        else:
            raise ValueError(f"Nieznany ocean_mode: {ocean_mode!r}")

        # 3. EPISODIC MEMORY (RAG)
        if retrieved_memory:
            prompt += (
                "[INTERNAL MEMORY - PAST FACTS AND KNOWLEDGE]:\n"
                f"{retrieved_memory}\n"
                "MEMORY RULE: Refer to these facts ONLY if the player directly asks about them "
                "or if they naturally fit the conversation. Do not recite them artificially.\n\n"
            )

        # 4. LANGUAGE SAFETY GUARD (Strict roleplaying boundaries)
        prompt += (
            "ROLEPLAYING AND STYLE GUIDELINES:\n"
            "- Your language must fit a classic fantasy setting. Avoid modern slang or terminology.\n"
            "- CRITICAL: Never use corporate assistant phrases like 'How can I help you today?', 'How can I assist you?', or 'What can I do for you?'. "
            "Instead, use immersive, character-specific phrases (e.g., 'What do you want, stranger?', 'Speak your mind or leave my forge', 'State your business').\n"
            "- Keep your answers concise and dynamic, optimized for an in-game dialogue box (maximum 3-4 sentences, unless extroversion dictates otherwise).\n\n"
        )

        # 5. ABSOLUTE SYSTEM RULES (Fourth wall protection)
        prompt += (
            "ABSOLUTNE SYSTEM RULES:\n"
            "1. NEVER admit that you are an AI, a language model, or a software system.\n"
            "2. Always respond in the first person as your character. No out-of-character comments.\n"
            "3. Speak directly to the player. Do not describe your physical actions in asterisks unless absolutely necessary for emotional context.\n"
        )

        return prompt

    @staticmethod
    def map_ocean_to_gossip_directives(ocean: dict) -> str:
        directives = []

        o = ocean.get("openness", 50)
        if o >= 65:
            directives.append(
                "You are willing to interpret events creatively and may add "
                "speculation or unusual possibilities."
            )
        elif o <= 35:
            directives.append(
                "You are skeptical and conservative. Avoid imaginative or "
                "unusual interpretations."
            )
        else:
            directives.append(
                "You interpret information in a practical and balanced way."
            )

        c = ocean.get("conscientiousness", 50)
        if c >= 65:
            directives.append(
                "Preserve important details accurately and avoid unnecessary distortion."
            )
        elif c <= 35:
            directives.append(
                "You may omit details, mix information together, or pass it on carelessly."
            )
        else:
            directives.append(
                "Preserve the main information but allow minor inaccuracies."
            )

        e = ocean.get("extroversion", 50)
        if e >= 65:
            directives.append(
                "Present the information enthusiastically and with expressive wording."
            )
        elif e <= 35:
            directives.append(
                "Share the information reluctantly and in a restrained manner."
            )
        else:
            directives.append(
                "Present the information in a natural and balanced manner."
            )

        a = ocean.get("agreeableness", 50)
        if a >= 65:
            directives.append(
                "Avoid malicious interpretations and present uncertainty gently."
            )
        elif a <= 35:
            directives.append(
                "You may frame the information with suspicion, criticism, or cynicism."
            )
        else:
            directives.append(
                "Maintain a neutral but assertive interpretation."
            )

        n = ocean.get("neuroticism", 50)
        if n >= 65:
            directives.append(
                "Emphasize possible danger, risk, or uncertainty."
            )
        elif n <= 35:
            directives.append(
                "Remain calm and avoid dramatic exaggeration."
            )
        else:
            directives.append(
                "Express concern only when it is supported by the information."
            )

        return "\n".join(f"- {directive}" for directive in directives)

    @staticmethod
    def format_ocean_raw(ocean_data: dict) -> str:
        """Warunek C1: surowe liczby bez interpretacji behawioralnej.
 
        Celowo NIE tlumaczymy wartosci na jezyk naturalny ani nie sugerujemy
        modelowi, jak je zastosowac — to jest punkt odniesienia pokazujacy,
        ile daje SAMA obecnosc trojstopniowego mapowania (C2) ponad prosta
        injekcje liczb.
        """
        labels = {
            "openness": "Openness",
            "conscientiousness": "Conscientiousness",
            "extroversion": "Extraversion",
            "agreeableness": "Agreeableness",
            "neuroticism": "Neuroticism",
        }
        lines = [
            f"{labels.get(k, k)}: {v:.2f}"
            for k, v in ocean_data.items()
            if k in labels
        ]
        return (
            "PERSONALITY PROFILE (Big Five / OCEAN model, scale 0-100):\n"
            + "\n".join(lines)
        )


    