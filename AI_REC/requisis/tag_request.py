import aiohttp
import asyncio
import async_timeout
import csv
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY não encontrada. Insira no .env")

BASE_URL = "https://api.rawg.io/api/tags"
CSV_PATH = "tags.csv"

async def collect_tags(page_size=40, output_file=CSV_PATH):
    fields = ["id", "name", "slug", "games_count", "image_background"]
    all_tags = []

    async def fetch_page(session, page):
        params = {
            "key": API_KEY,
            "page": page,
            "page_size": page_size
        }
        for attempt in range(3):
            try:
                async with async_timeout.timeout(30):
                    async with session.get(BASE_URL, params=params) as resp:
                        if resp.status == 200:
                            return await resp.json()
                        else:
                            print(f"Atenção: Status {resp.status}, tentativa {attempt + 1}/3")
            except Exception as e:
                print(f"Erro na requisição: {e}, tentativa {attempt + 1}/3")
            await asyncio.sleep(2)
        print(f"Falha ao coletar a página {page}")
        return None

    async with aiohttp.ClientSession() as session:
        # Pega a primeira página para saber o total
        first_page = await fetch_page(session, 1)
        if not first_page or "count" not in first_page:
            print("Não foi possível obter o número total de tags disponíveis.")
            return

        total_disponiveis = first_page["count"]
        print(f"📊 Total de tags disponíveis na API: {total_disponiveis}")

        num_paginas = (total_disponiveis + page_size - 1) // page_size

        # Processa a primeira página
        all_tags.extend(first_page.get("results", []))

        # Processa páginas restantes
        pages_data = [fetch_page(session, page) for page in range(2, num_paginas + 1)]
        for future in asyncio.as_completed(pages_data):
            data = await future
            if not data or "results" not in data:
                continue
            all_tags.extend(data["results"])

        # Ordena pelo número de jogos que usam a tag (games_count) - mais populares primeiro
        all_tags.sort(key=lambda x: x["games_count"], reverse=True)

        # Salva em CSV
        with open(output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for tag in all_tags:
                writer.writerow({k: str(tag.get(k, "")) for k in fields})

    print(f"✅ Coleta finalizada! {len(all_tags)} tags salvas em '{output_file}'")

if __name__ == "__main__":
    asyncio.run(collect_tags())
