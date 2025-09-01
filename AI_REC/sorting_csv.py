import csv

# NÃO MODIFIQUE NADA CASO NÃO TENHA PERMISSÃO. Vá para a linha 74 e modifique os parâmetros que deseja, e execute.
# Use para ordenar depois de puxar os dados da API 
# Recomendação: Puxe os mais populares e mais avaliados pela API; Depois, ordene como preferir aqui
def ordenar_csv_stream(input_file, output_file=None, criterios=None, descendente=None):
    """
    Ordena um CSV de forma eficiente, mesmo para arquivos grandes.

    :param input_file: caminho do CSV de entrada
    :param output_file: caminho do CSV de saída (se None, sobrescreve input_file)
    :param criterios: lista de campos para ordenar, ex: ["nota_media", "nome"]
    :param descendente: lista de bools indicando ordem descendente por critério, ex: [True, False]
    """
    if criterios is None:
        criterios = ["nome"]

    if descendente is None:
        descendente = [False] * len(criterios)
    
    if output_file is None:
        output_file = input_file

    # Criando chave de ordenação segura
    def chave(jogo):
        chaves = []
        for crit, desc in zip(criterios, descendente):
            val = jogo.get(crit, "")
            
            if crit == "nota_media":
                val = float(val)
            
            elif crit == "ano_lancamento":
                val = int(val)
            
            # Inverte se descendente
            chaves.append(-val if isinstance(val, (int, float)) and desc else val)
        
        return tuple(chaves)

    # Ler e converter CSV em um gerador
    def gerar_jogos():
        with open(input_file, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # Conversão com valor padrão se inválido
                try:
                    row["nota_media"] = float(row["nota_media"])
                except (ValueError, TypeError, KeyError):
                    row["nota_media"] = 0.0
                
                try:
                    row["ano_lancamento"] = int(row["ano_lancamento"])
                except (ValueError, TypeError, KeyError):
                    row["ano_lancamento"] = 0
                
                yield row

    jogos_ordenados = sorted(gerar_jogos(), key=chave)

    # Escrever CSV
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=jogos_ordenados[0].keys())
        writer.writeheader()
        writer.writerows(jogos_ordenados)

    print(f"✅ CSV ordenado salvo em '{output_file}'")






# MODIFIQUE OS PARAMS AQUI
ordenar_csv_stream(input_file='jogos.csv', output_file='jogos_alfabetic.csv',
    criterios=["nome", "ano_lancamento"],
    descendente=[False, False]
)
