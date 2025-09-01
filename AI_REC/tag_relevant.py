import pandas as pd

tags_df = pd.read_csv('tags.csv')

# Função para classificar peso baseado em relevância aparente
def assign_weight(tag_name):
    tag_name_lower = tag_name.lower()
    
    # Tags irrelevantes / genéricas
    low_relevance = ['Steam Achievements', 'Full controller support', 'Early Access', 'controller support', 'engine', 'fun', 
                     'Atmospheric', 'colorful', 'cinematic', 'moddable', 'hdr', 'Steam Leaderboards', 'steam-trading-cards', 'Great Soundtrack'
                     'collectibles', 'commentary', 'lego', 'games workshop', "Steam Cloud", "Family Sharing", 'Game Jam', "Game Maker's ToolKit Jam",
                     'My First Game Jam', 'achievemens', 'In-App Purchase', 'Soundtrack', 'Steam Workshop', 'Mod', 'VR Only', 'Kickstarter', 'leaderboards',
                     'Game Boy Advance', 'HDR available', 'Steam Turn Notifications', 'Valve Anti-Cheat enabled', 'Includes Sources SDK'
                     'Unity', 'Unreal Engine', 'Short', 'Funny', 'Controller', 'Partial Controller Support'
                     
                     
                     ]
    if any(word in tag_name_lower for word in low_relevance):
        return 0.5
    
    # Tags medianamente relevantes
    medium_relevance = ['Adventure', 'Singleplayer', 'co-op', 'Multiplayer', 'GameMaker', 'RPGMaker', 'Fishing', 'NES', 'Villain Protagonist',
                        'puzzle', 'Minigames', 'strategy', 'racing', '2D', '3D', 'Open World', 'PvE', 'Perma Death', 'Demake', 'Remake'
                        'Pixel Graphics', 'Third Person', 'VR', 'Romance', 'Isometric', 'Dungeons & Dragons', 'Nintendo DS',
                        'Boss Rush', 'Nintendo 64', 'Retro', 'Cute', 'First-Person', 'Fantasy', 'Exploration', 'Anime', 'Mystery', 'Endless',
                        'Difficult', 'PvP', 'Female Protagonist', ''

                        
                        
                        
                        ]
    if any(word in tag_name_lower for word in medium_relevance):
        return 2
    
    # Tags mais relevantes (definem gameplay principal)
    high_relevance = ['action', 'RPG', 'FPS', "Shoot'Em Up", 'Visual Novel', 'Sandbox', 'Roguelite', 'Hack and Slash', 'Card Game', 'Football',
                      'shooter', 'platformer', 'fighting', 'Horror', 'Survival', 'simulator', 'Dating Sim', 'JRPG', "Beat'em up", 'Life Sim',
                      'sports', 'sandbox', 'Stealth', 'Roguelike', 'Survival Horror', 'Puzzle Platformer', 'Metroidvania', 'Board Game', '3D Plataformer'
                      'Party Game', 'Soccer', 'Golf', 'Life Simulation', 'MMORPG', 'tycoon', 'Competitive', '3D Fighter', '2D Fighter', 'Pinball', 
                      'Hero Shooter', 'Battle Royale', 'MOBA', 'Trading Card Game', 'LoveCraftian Horror', 'Naval', 'LEGO', 'Job Simulator', 'Music',
                      'Point & Click', 'Bullet Hell', 'skateboarding'

                      
                      
                      ]
    if any(word in tag_name_lower for word in high_relevance):
        return 3
    
    # Default médio
    return 1.5

# Criar dicionário de tags com pesos
tags_pesos_dynamic = {row['name']: assign_weight(row['name']) for _, row in tags_df.iterrows()}

def get_tags_weight():
    return list(list(tags_pesos_dynamic.items())[:])
# Mostrar algumas entradas

