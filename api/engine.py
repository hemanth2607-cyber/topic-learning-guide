import os
from openai import OpenAI

class GrokEngine:
    def __init__(self):
        self.api_key = os.getenv("GROK_API_KEY")
        if not self.api_key:
            raise ValueError("GROK_API_KEY environment variable is not set.")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1" 
        )

    def _get_prompts(self, level: str, topic: str):
        tag_instruction = (
            "\\n\\nCRITICAL FORMATTING & LANGUAGE REQUIREMENTS:\\n"
            "1. NO PREAMBLE RULE: Do NOT write any introduction, greetings, conversational filler, "
            "or metadata explanations. Start your output immediately with '[EXPLANATION_SECTION]'.\\n"
            "2. LANGUAGE DETECT RULE: Detect the language of the input topic. "
            "Write the content under all sections in that exact language/script.\\n"
            "3. THANGLISH QUALITY RULE: If the detected language is Thanglish, you must write in natural, "
            "colloquial phonetic Tamil used by native speakers. Follow these examples to avoid artificial translations:\\n"
            "   - Improper/Bad Thanglish: 'Object-oriented programming paradigm based on conceptual objects. Idhu code structures direct panna help pannum.'\\n"
            "   - Proper/Natural Thanglish (Use this style): 'Object-oriented programming (OOP) oru programming paradigm. Idhu full-ah 'objects' concepts-ah base panni dhaan work aagum. Real-world structures-ah code-la mimic panna idhu romba use aagum.'\\n"
            "   - Key Vocabulary: Use conversational words like 'idhu', 'iruku', 'solli tharum', 'kudunga', 'neriya concepts', 'romba useful-ah irukum', 'epdi work aagum-nu paakalam'.\\n"
            "4. TAG RULE: Keep the section tags exactly in English (e.g., [EXPLANATION_SECTION]) so the backend parser can read them.\\n\\n"
            "[EXPLANATION_SECTION]\\n"
            "Your explanation text here...\\n\\n"
            "[MODULES_SECTION]\\n"
            "Your modules list here...\\n\\n"
            "[RESOURCES_SECTION]\\n"
            "Your resources list here...\\n"
        )

        if level == "Overview":
            system = (
                "You are an academic guide. Create an overview learning path.\\n"
                "- Under '[EXPLANATION_SECTION]': Provide a simple 1-paragraph summary of the topic.\\n"
                "- Under '[MODULES_SECTION]': Outline 3 basic, core modules to learn.\\n"
                "- Under '[RESOURCES_SECTION]': Recommend 2 highly accessible basic books/guides, indicating format (hard copy, e-book) and where to find them."
                + tag_instruction
            )
            user = f"Generate an Overview path for '{topic}'."
        elif level == "Detailed":
            system = (
                "You are an academic instructor. Create a detailed learning path.\\n"
                "- Under '[EXPLANATION_SECTION]': Provide a thorough explanation of the concepts so the user can easily explain it to others.\\n"
                "- Under '[MODULES_SECTION]': Outline 5-6 structured modules to master.\\n"
                "- Under '[RESOURCES_SECTION]': Recommend 3-4 essential textbooks (with authors and formats) and high-quality web resources."
                + tag_instruction
            )
            user = f"Generate a Detailed path for '{topic}'."
        else:  # Deep Learn
            system = (
                "You are an advanced researcher and professor. Create a deep research-level learning path.\\n"
                "- Under '[EXPLANATION_SECTION]': Provide an academic breakdown covering advanced theories, histories, and current research trends.\\n"
                "- Under '[MODULES_SECTION]': Outline an exhaustive syllabus of advanced concepts and mathematical/design foundations.\\n"
                "- Under '[RESOURCES_SECTION]': Recommend graduate-level textbooks, seminal academic papers, and advanced online courses."
                + tag_instruction
            )
            user = f"Generate a Deep Learn path for '{topic}'."

        return system, user

    def _parse_tags(self, text: str) -> dict:
        """Robust string splitting using native python find methods instead of regular expressions."""
        result = {
            "explanation": "",
            "modules": "",
            "resources": ""
        }

        exp_tag = "[EXPLANATION_SECTION]"
        mod_tag = "[MODULES_SECTION]"
        res_tag = "[RESOURCES_SECTION]"

        # Locate tag indexes inside the response
        exp_idx = text.find(exp_tag)
        mod_idx = text.find(mod_tag)
        res_idx = text.find(res_tag)

        # 1. Parse Explanation
        if exp_idx != -1:
            end_idx = len(text)
            if mod_idx != -1 and mod_idx > exp_idx:
                end_idx = min(end_idx, mod_idx)
            if res_idx != -1 and res_idx > exp_idx:
                end_idx = min(end_idx, res_idx)
            result["explanation"] = text[exp_idx + len(exp_tag):end_idx].strip()

        # 2. Parse Modules
        if mod_idx != -1:
            end_idx = len(text)
            if exp_idx != -1 and exp_idx > mod_idx:
                end_idx = min(end_idx, exp_idx)
            if res_idx != -1 and res_idx > mod_idx:
                end_idx = min(end_idx, res_idx)
            result["modules"] = text[mod_idx + len(mod_tag):end_idx].strip()

        # 3. Parse Resources
        if res_idx != -1:
            end_idx = len(text)
            if exp_idx != -1 and exp_idx > res_idx:
                end_idx = min(end_idx, exp_idx)
            if mod_idx != -1 and mod_idx > res_idx:
                end_idx = min(end_idx, mod_idx)
            result["resources"] = text[res_idx + len(res_tag):end_idx].strip()

        # Fallback if no tags are matched
        if not result["explanation"] and not result["modules"] and not result["resources"]:
            result["explanation"] = text
            result["modules"] = "Review the explanation tab for details."
            result["resources"] = "Review the explanation tab for details."

        return result

    def generate_path(self, topic: str, level: str) -> dict:
        system_prompt, user_prompt = self._get_prompts(level, topic)
        
        response = self.client.chat.completions.create(
            model="llama-3.3-70b-versatile", 
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        
        raw_content = response.choices[0].message.content
        return self._parse_tags(raw_content)