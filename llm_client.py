import os, requests

class LLMClient:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.hf_key = os.getenv("HF_TOKEN")  # Hugging Face token (free to generate)

    def generate(self, prompt: str) -> str:
        if self.openai_key:
            return self._generate_openai(prompt)
        return self._generate_huggingface(prompt)

    def _generate_openai(self, prompt: str) -> str:
        import openai
        openai.api_key = self.openai_key
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}]
        )
        return resp["choices"][0]["message"]["content"]

    def _generate_huggingface(self, prompt: str) -> str:
        if not self.hf_key:
            raise ValueError("No Hugging Face token found. Set HF_TOKEN env variable.")

        url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        headers = {"Authorization": f"Bearer {self.hf_key}"}
        payload = {"inputs": prompt, "parameters": {"max_new_tokens": 150}}

        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()[0]["generated_text"]