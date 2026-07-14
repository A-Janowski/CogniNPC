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
    def _map_ocean_to_directives(ocean: dict) -> str:
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
    def build_system_prompt(profile: dict, retrieved_memory: str) -> str:
        """
        Assembles the final English System Prompt sent to the LLM.
        Includes safety guards to prevent default AI assistant patterns.
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
        prompt += "BEHAVIORAL STYLE (Strictly adhere to these traits):\n"
        prompt += PromptBuilder._map_ocean_to_directives(ocean_data)
        prompt += "\n\n"

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