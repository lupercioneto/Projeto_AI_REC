import aiohttp
import asyncio
import csv
import os
import json
from dotenv import load_dotenv
from typing import Dict, List

# Carregar API key
load_dotenv()
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    raise SystemExit("Erro: Configure API_KEY no arquivo .env")

CACHE_DIR = "cache"
os.makedirs(CACHE_DIR, exist_ok=True)

SEM = asyncio.Semaphore(15)  # Limite de requisições simultâneas
PAGE_SIZE = 40  # Quantos jogos pegar por página

async def fetch_json(session: aiohttp.ClientSession, url: str, cache_file: str, timeout: int = 10) -> dict:
    """Busca JSON da URL, usando cache se disponível."""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass  # Ignora cache corrompido

    try:
        async with SEM:
            async with session.get(url, timeout=timeout) as response:
                response.raise_for_status()
                data = await response.json()
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                return data
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        print(f"Erro na requisição {url}: {e}")
        return {}

async def fetch_game_individual(session: aiohttp.ClientSession, game_id: int) -> Dict[str, any]:
    """Busca dados do jogo individualmente (apenas se não tiver no cache)."""
    cache_file = os.path.join(CACHE_DIR, f"{game_id}_game.json")
    url = f"https://api.rawg.io/api/games/{game_id}?key={API_KEY}&screenshots=true"
    data = await fetch_json(session, url, cache_file)
    game_name = data.get('name', f"Unknown (ID {game_id})")
    screenshot_urls = [shot['image'] for shot in data.get('short_screenshots', [])[:5]]
    return {'id': game_id, 'name': game_name, 'screenshot_urls': screenshot_urls}

async def fetch_games_from_list(session: aiohttp.ClientSession, page: int = 1) -> List[Dict[str, any]]:
    """Busca uma página de jogos da API e retorna os dados básicos."""
    cache_file = os.path.join(CACHE_DIR, f"page_{page}.json")
    url = f"https://api.rawg.io/api/games?key={API_KEY}&page={page}&page_size={PAGE_SIZE}"
    data = await fetch_json(session, url, cache_file)
    games = []
    for game in data.get('results', []):
        game_id = game['id']
        game_name = game.get('name', f"Unknown (ID {game_id})")
        screenshot_urls = [shot['image'] for shot in game.get('short_screenshots', [])[:5]] if 'short_screenshots' in game else []
        games.append({'id': game_id, 'name': game_name, 'screenshot_urls': screenshot_urls})
    return games

async def main():
    # Ler IDs do CSV
    try:
        with open('jogos.csv', 'r', encoding='utf-8') as f:
            game_ids = [int(row['id']) for row in csv.DictReader(f) if row.get('id', '').isdigit()]
    except FileNotFoundError:
        raise SystemExit("Arquivo 'jogos.csv' não encontrado!")

    async with aiohttp.ClientSession() as session:
        results = []

        # 1️⃣ Primeiro tentamos pegar do cache ou de páginas já baixadas
        for page in range(1, (len(game_ids) // PAGE_SIZE) + 2):
            page_games = await fetch_games_from_list(session, page)
            results.extend(page_games)

        # 2️⃣ Agora garantimos que todos os IDs do CSV foram atendidos
        existing_ids = {game['id'] for game in results}
        missing_ids = [gid for gid in game_ids if gid not in existing_ids]

        if missing_ids:
            print(f"Buscando {len(missing_ids)} jogos individuais que não estavam nas páginas...")
            tasks = [fetch_game_individual(session, gid) for gid in missing_ids]
            individual_results = await asyncio.gather(*tasks)
            results.extend(individual_results)

        # Filtrando apenas os jogos do CSV original
        final_results = [game for game in results if game['id'] in game_ids]

    # Salvar CSV final
    with open('games_with_screenshots.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'name', 'screenshot_urls'])
        writer.writeheader()
        for game in final_results:
            writer.writerow({
                'id': game['id'],
                'name': game['name'],
                'screenshot_urls': json.dumps(game['screenshot_urls'], ensure_ascii=False)
            })

    print("Processamento concluído! Requisições otimizadas e cache utilizado.")

if __name__ == "__main__":
    asyncio.run(main())
