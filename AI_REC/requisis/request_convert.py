import aiohttp
import asyncio
import async_timeout
import csv
from datetime import datetime
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY não encontrada. Insira no .env")

# URL Base
BASE_URL = "https://api.rawg.io/api/games"
CSV_PATH = "jogos.csv"


# Request pra coletar jogos
async def colect_games(total_games=1500, page_size=40, output_file=CSV_PATH, order_by="-rating,-released"):
    fields = ["id", "nome", "tags", "generos", "nota_media", "metacritic", "ano_lancamento"]
    ids_seens = set()
    games_colected = 0

    async def fetch_page(session, page):
        params = {
            "key": API_KEY,
            "page": page,
            "page_size": page_size,
            "ordering": order_by
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
        first_page = await fetch_page(session, 1)
        
        if not first_page or "count" not in first_page:
            print("Não foi possível obter o número total de jogos disponíveis.")
            return

        total_disponiveis = first_page["count"]
        print(f"📊 Total de jogos disponíveis na API: {total_disponiveis}")

        # Ajuste do total de coleta
        total_games = min(total_games, total_disponiveis)
        num_paginas = (total_games + page_size - 1) // page_size
        print(f"ℹ️ Coleta de {total_games} jogos em {num_paginas} páginas.")

        # CSV
        with open(output_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for jogo in first_page.get("results", []):
                if jogo["id"] in ids_seens:
                    continue
                ids_seens.add(jogo["id"])

                ano = None
                if jogo.get("released"):
                    try:
                        ano = datetime.strptime(jogo["released"], "%Y-%m-%d").year
                    except ValueError:
                        pass

                jogo_dict = {
                    "id": jogo["id"],
                    "nome": jogo["name"],
                    "tags": [t["name"] for t in jogo.get("tags", [])],
                    "generos": [g["name"] for g in jogo.get("genres", [])],
                    "nota_media": jogo.get("rating", 0.0),
                    "metacritic": jogo.get("metacritic", ""),
                    "ano_lancamento": ano
                    
                }

                writer.writerow({k: str(v) if v is not None else "" for k, v in jogo_dict.items()})
                games_colected += 1
                if games_colected >= total_games:
                    print(f"✅ Limite de {total_games} jogos alcançado!")
                    return

            # Agora processa as páginas restantes
            pages_data = [fetch_page(session, page) for page in range(2, num_paginas + 1)]
            for future in asyncio.as_completed(pages_data):
                data = await future
                if not data or "results" not in data:
                    continue

                for jogo in data["results"]:
                    if jogo["id"] in ids_seens:
                        continue
                    ids_seens.add(jogo["id"])

                    ano = None
                    if jogo.get("released"):
                        try:
                            ano = datetime.strptime(jogo["released"], "%Y-%m-%d").year
                        except ValueError:
                            pass

                    jogo_dict = {
                        "id": jogo["id"],
                        "nome": jogo["name"],
                        "tags": [t["name"] for t in jogo.get("tags", [])],
                        "generos": [g["name"] for g in jogo.get("genres", [])],
                        "nota_media": jogo.get("rating", 0.0),
                        "metacritic": jogo.get("metacritic", ""),
                        "ano_lancamento": ano
                    }

                    writer.writerow({k: str(v) if v is not None else "" for k, v in jogo_dict.items()})
                    games_colected += 1

                    if games_colected >= total_games:
                        print(f"✅ Limite de {total_games} jogos alcançado!")
                        return

    print(f"✅ Coleta finalizada! {games_colected} jogos salvos em '{output_file}'")


# Seção de Screenshots
# - Fetch
async def fetch_screenshots(session, game_id):
    url = f"{BASE_URL}/{game_id}/screenshots"
    
    params = {
        "key": API_KEY,
        "page_size": 5
    }

    for attempt in range(3):  # até 3 tentativas
        try:
            async with async_timeout.timeout(15):
                async with session.get(url, params=params) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return [shot["image"] for shot in data.get("results", [])]
                    else:
                        print(f"⚠️ Status {resp.status} para jogo {game_id}, tentativa {attempt+1}/3")
        except Exception as e:
            print(f"❌ Erro na requisição para jogo {game_id}: {e}, tentativa {attempt+1}/3")
        await asyncio.sleep(2)

    print(f"❌ Falha ao coletar screenshots do jogo {game_id}")
    return []


# Coletar Screenshots
async def collect_screenshots_for_games(game_ids):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_screenshots(session, game_id) for game_id in game_ids]
        results = await asyncio.gather(*tasks)
        screenshots_by_game = dict(zip(game_ids, results))
        return screenshots_by_game


async def show_screenshots():
    games_ids = [258322, 9767, 292844]
    screenshots = await collect_screenshots_for_games(games_ids)
    
    for game_id, shots in screenshots.items():
        print(f"🎮 Jogo {game_id}:")
        for idx, shot in enumerate(shots, start=1):
            print(f"  {idx}: {shot}")

if __name__ == "__main__":
    # asyncio.run(colect_games(total_games=10000, page_size=50))
    asyncio.run(show_screenshots())