import os
# pyrefly: ignore [missing-import]
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
            "2. LANGUAGE MATCHING RULE:\\n"
            "   - If the user's input topic is in standard English, you MUST generate the entire output in clean, standard English.\\n"
            "   - If the user's input topic is in Tamil script (e.g. தமிழ்), you MUST generate the entire output in clean, standard Tamil script.\\n"
            "   - If the user's input topic is in Thanglish (e.g. 'snake pathi sollu'), you MUST generate the entire output in natural, conversational Thanglish (using phonetic Tamil words like 'solren', 'iruku', 'kudunga').\\n"
            "3. TARGETED SEARCH LINKING RULE:\\n"
            "   - AUTHOR & CHANNEL REQUIREMENT: For every book, you MUST write the author's name. For every tutorial, you MUST specify the exact YouTube channel name (e.g., Programming with Mosh, freeCodeCamp) or platform (e.g., W3Schools, GeeksforGeeks).\\n"
            "   - NO BROKEN LINKS RULE: To prevent 404 or broken links, do NOT guess deep URLs. Instead, construct high-precision search queries using these exact templates (replace spaces with '+' symbols inside the URLs):\\n"
            "     * For exact YouTube Videos: Link to a targeted search query on YouTube: 'https://www.youtube.com/results?search_query=Topic+Tutorial+by+ChannelName'. "
            "Example: '[OOP Tutorial by Programming with Mosh 🔗](https://www.youtube.com/results?search_query=OOP+Tutorial+by+Programming+with+Mosh)'.\\n"
            "     * For exact Web Tutorials: Use Google's 'site:' operator to target only the official verified site. "
            "Example: '[Java OOP Concepts (GeeksforGeeks) 🔗](https://www.google.com/search?q=site:geeksforgeeks.org+Java+OOP+Concepts)'.\\n"
            "     * For Books (E-books): Link to Google Books search query: 'https://www.google.com/search?tbm=bks&q=Book+Title+Author+Name'. "
            "Example: '[Head First OOP by Brett McLaughlin 🔗](https://www.google.com/search?tbm=bks&q=Head+First+OOP+Brett+McLaughlin)'.\\n"
            "   - For physical/offline resources, do NOT create a hyperlink. "
            "Instead, write: 'Book Title by Author [📖 Physical Copy Only]'.\\n"
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
                "- Under '[MODULES_SECTION]': Outline 3 basic, core modules to learn, providing a high-precision YouTube video search link for each.\\n"
                "- Under '[RESOURCES_SECTION]': Recommend 2 highly accessible basic books/guides, using targeted Google Books search links."
                + tag_instruction
            )
            user = f"Generate an Overview path for '{topic}'."
        elif level == "Detailed":
            system = (
                "You are an academic instructor. Create a detailed learning path.\\n"
                "- Under '[EXPLANATION_SECTION]': Provide a thorough explanation of the concepts so the user can easily explain it to others.\\n"
                "- Under '[MODULES_SECTION]': Outline 5-6 structured modules to master, linking each to a targeted YouTube tutorial or a site-specific Google search.\\n"
                "- Under '[RESOURCES_SECTION]': Recommend 3-4 essential textbooks with authors, formats, and targeted search links."
                + tag_instruction
            )
            user = f"Generate a Detailed path for '{topic}'."
        else:  # Deep Learn
            system = (
                "You are an advanced researcher and professor. Create a deep research-level learning path.\\n"
                "- Under '[EXPLANATION_SECTION]': Provide an academic breakdown covering advanced theories, histories, and current research trends.\\n"
                "- Under '[MODULES_SECTION]': Outline an exhaustive syllabus of advanced concepts, linking each module to targeted academic search queries or official documentations.\\n"
                "- Under '[RESOURCES_SECTION]': Recommend graduate-level textbooks, seminal academic papers, and advanced online courses with verified search queries."
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
                end_idx = min(end_idx, res_idx)
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