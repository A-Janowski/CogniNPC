from pathlib import Path
from loguru import logger
import yaml
import time
from app.services.chromadb_service import ChromaDbService
from app.services.ollama_service import OllamaService
from app.core.prompt_builder import PromptBuilder
from app.core.config import settings

DEFAULT_RAG_K = 2

class NPCService:
    def __init__(self):
        self.memory = ChromaDbService()
        self.llm = OllamaService()
        self.prompt_builder = PromptBuilder()
        self.profile_dir = Path(settings.DATA_DIR) / "npc_profiles"

    def process_chat(
        self,
        npc_id: str,
        player_message: str,
        model: str | None = None,
        rag_k: int | None = None,
        num_predict: int | None = None,
        skip_memory_write: bool = False,
    ) -> tuple[str, bool, dict[str, object]]:
        
        t_start = time.perf_counter()
        effective_k = DEFAULT_RAG_K if rag_k is None else rag_k
        
        # 1. Loading NPC profile
        logger.debug(f"Loading NPC profile for ID: {npc_id}")
        profile = self.prompt_builder.load_npc_profile(npc_id)
        
        # 2. Searching for relevant context in memory (RAG)
        logger.debug(f"Retrieving context for NPC ID: {npc_id} with player message: {player_message}, limit: {effective_k}")
        context, rag_metrics = self.memory.retrieve_context(
            npc_id, player_message, limit=effective_k
        )
        
        # 3. Building the prompt
        logger.debug(f"Building system prompt for NPC ID: {npc_id}")
        t_prompt0 = time.perf_counter()
        system_prompt = self.prompt_builder.build_system_prompt(profile, context)
        full_prompt = f"{system_prompt}\n\nPlayer: {player_message}\nYou:"
        prompt_build_ms = (time.perf_counter() - t_prompt0) * 1000
        
        # 4. Inference
        logger.debug(f"Generating response for NPC ID: {npc_id}, model: {model}, num_predict: {num_predict}")
        llm_result = self.llm.generate(full_prompt, model=model, num_predict=num_predict)
        response_text = llm_result.get("response_text", "")
        
        # 5. Save the current conversation to memory
        logger.debug(f"Saving conversation for NPC ID: {npc_id}")
        if not skip_memory_write:
            self.memory.add_memory(
                npc_id,
                f"Player said: {player_message}. I responded: {response_text}",
                "conversation"
            )

        used_memory = len(context) > 0
        total_wall_ms = (time.perf_counter() - t_start) * 1000
        logger.info(f"Successfully processed chat for {npc_id}. Memory used: {used_memory}. Total time: {total_wall_ms:.2f} ms")

        # Orchestartion time = total time minus Ollama inference time.
        # ollama_total_ms includes the time spent in the LLM engine
        # (already contains load+prefill+decode), so it's a fair comparison of
        # "cost of my code" vs "cost of the LLM engine".
        ollama_total_ms = llm_result.get("load_duration", 0) / 1e6 \
            + llm_result.get("prompt_eval_duration", 0) / 1e6 \
            + llm_result.get("eval_duration", 0) / 1e6
 
        metrics = dict(llm_result)
        metrics.update(rag_metrics)
        metrics["stage_prompt_build_ms"] = round(prompt_build_ms, 3)
        metrics["orchestration_ms"] = round(total_wall_ms - ollama_total_ms, 3)
        metrics["wall_total_ms"] = round(total_wall_ms, 3)
        metrics["rag_k_requested"] = effective_k
 
        return response_text, used_memory, metrics
    
    def get_all_npc_ids(self):
        return sorted([
            file.stem
            for file in self.profile_dir.glob("*.yaml")
        ])
    
    def get_npc_profile(self, npc_id: str):
        profile_path = self.profile_dir / f"{npc_id}.yaml"
        if not profile_path.exists():
            raise FileNotFoundError(f"NPC '{npc_id}' not found.")
        with open(profile_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def process_gossip(self, npc_source: str, npc_target: str) -> tuple[bool, str]:
        source_memory = self.memory.get_random_memory(npc_source)
        if not source_memory:
            logger.trace(f"Source NPC '{npc_source}' has no memories to share.")
            return False, "Source NPC has no memories to share."
        
        logger.trace(f"Source NPC '{npc_source}' memory retrieved for gossip: {source_memory}")
        source_profile = self.prompt_builder.load_npc_profile(npc_source)
        mutation_prompt = (
            self.build_gossip_mutation_prompt(
                source_profile,
                source_memory
            )
        )

        llm_result = self.llm.generate(mutation_prompt)
        gossip_text = llm_result.get("response_text", "")
        self.memory.add_memory(
            npc_target,
            gossip_text,
            source=f"gossip_from_{npc_source}"
        )

        return True, gossip_text    


    @staticmethod
    def build_gossip_mutation_prompt(source_profile: dict, source_memory: str) -> str:
        name = source_profile.get("name", "Unknown")
        profession = source_profile.get("profession", "Unknown")
        ocean = source_profile.get("ocean", {})

        personality = PromptBuilder.map_ocean_to_gossip_directives(ocean)

        return f"""
            You simulate how an NPC passes information to another NPC in a fantasy RPG.

            SOURCE NPC:
            Name: {name}
            Profession: {profession}

            PERSONALITY:
            {personality}

            TASK:
            Transform the memory below into a short piece of information that this NPC
            might pass to another person.

            The NPC's personality must influence:
            - how confidently the information is presented,
            - whether details are preserved or distorted,
            - how much uncertainty or exaggeration is added,
            - the tone and wording of the statement.

            RULES:
            1. Preserve the core subject and main event.
            2. Do not invent completely unrelated people, places, events, or motives.
            3. Add only a small and believable amount of uncertainty, exaggeration,
            suspicion, or distortion.
            4. Do not claim that the information is certainly true unless the original
            memory explicitly confirms it.
            5. Keep the result to exactly one sentence.
            6. Use language appropriate for a classic fantasy setting.
            7. Return only the transformed rumor, without explanations, labels, or quotation marks.

            MEMORY:
            {source_memory}

            TRANSMITTED INFORMATION:
            """.strip()