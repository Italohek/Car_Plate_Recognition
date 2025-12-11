import cv2
import easyocr
from ultralytics import YOLO
import re
from collections import Counter

# path do video input e video output
video_path = 'videos/teste.mp4'
output_path = 'videos/teste_novo_output.mp4'

# definir como vai ser a placa final (utilizando padrão na)

def corrigir_placa(texto):
    """
    Suporta e corrige:
    1. Mercosul: LLLNLNN (Ex: ABC1D23)
    2. Antiga:   LLLNNNN (Ex: ABC1234)
    """
    # 1. Limpeza
    texto_limpo = re.sub(r'[^a-zA-Z0-9]', '', texto).upper()
    
    if len(texto_limpo) != 7:
        return None

    # Dicionários de correção
    dict_char_to_num = {'O': '0', 'I': '1', 'J': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'B': '8', 'Q': '0', 'D': '0'}
    dict_num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}

    lista_chars = list(texto_limpo)

    # --- 2. CORREÇÃO DAS POSIÇÕES COMUNS ---
    # Tanto Mercosul quanto Antiga têm:
    # - 3 Primeiros: Letras
    # - 4º digito (index 3): Número
    # - 2 Últimos: Números

    # Posições 0, 1, 2: TÊM que ser LETRAS
    for i in [0, 1, 2]:
        if lista_chars[i] in dict_num_to_char:
            lista_chars[i] = dict_num_to_char[lista_chars[i]]

    # Posição 3: TEM que ser NÚMERO
    if lista_chars[3] in dict_char_to_num:
        lista_chars[3] = dict_char_to_num[lista_chars[3]]

    # Posições 5, 6: TÊM que ser NÚMEROS
    for i in [5, 6]:
        if lista_chars[i] in dict_char_to_num:
            lista_chars[i] = dict_char_to_num[lista_chars[i]]

    # --- 3. DECISÃO DO ÍNDICE 4 (O Pivô) ---
    # Aqui decidimos se é Mercosul ou Antiga baseada no que temos
    
    char_meio = lista_chars[4]

    # Regexes para validação final
    regex_mercosul = re.compile(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$')
    regex_antiga   = re.compile(r'^[A-Z]{3}[0-9]{4}$')

    # Cenário A: O caractere já é uma Letra? -> Tenta validar como Mercosul
    if char_meio.isalpha():
        placa_teste = "".join(lista_chars)
        if regex_mercosul.match(placa_teste):
            return f"{placa_teste}"
        
        # Se falhou, talvez seja um número lido como letra (Ex: 'B' que era '8')
        if char_meio in dict_char_to_num:
            lista_chars[4] = dict_char_to_num[char_meio] # Vira número
            
    # Cenário B: O caractere é (ou virou) um Número? -> Tenta validar como Antiga
    if lista_chars[4].isdigit():
        placa_teste = "".join(lista_chars)
        if regex_antiga.match(placa_teste):
             return placa_teste # 
        
        # Se falhou, talvez seja uma letra lida como número (Ex: '0' que era 'O')
        if lista_chars[4] in dict_num_to_char:
            lista_chars[4] = dict_num_to_char[lista_chars[4]] # Vira letra
            # Testa Mercosul de novo
            placa_teste = "".join(lista_chars)
            if regex_mercosul.match(placa_teste):
                return placa_teste # RETORNE APENAS A STRING PURA (7 chars)

    return None


print("Iniciando...")
model = YOLO('runs/detect/train7/weights/best.pt')  # carrega o modelo treinado
reader = easyocr.Reader(['en'], gpu=True) # 'en' reconhece melhor letras/numeros gerais que 'pt'

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# buffer de memoria utilizado para votar em quais predicões são as mais corretas
# estrutura: { track_id: [lista_de_leituras] }
memory_buffer = {}

while True:
    ret, frame = cap.read()
    if not ret: break

    # vamos rastrear o objeto entre os frames
    results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")

    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.id is not None:
                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # só vai processar se a caixa for grande o suficiente (placa muito longe dá erro)
                if (x2-x1) > 60 and (y2-y1) > 20:
                    
                    # 1. Recorte
                    crop = frame[y1:y2, x1:x2]

                    # 2. Upscaling (Aumentar resolução)
                    # Aumentamos 4x usando INTER_CUBIC ou INTER_LANCZOS4 (melhor qualidade)
                    crop_big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)

                    # 3. Escala de Cinza
                    gray = cv2.cvtColor(crop_big, cv2.COLOR_BGR2GRAY)

                    # 4. Aumento de Contraste e Brilho (CLAHE)
                    # Isso ajuda MUITO em dias nublados ou sombra (como na sua foto)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    gray = clahe.apply(gray)

                    # 5. Binarização (Preto e Branco Puro)
                    # O Otsu calcula o limiar ideal automaticamente.
                    # Isso remove o fundo cinza/azul da placa e deixa só as letras pretas.
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # 6. (O PULO DO GATO) Adicionar Borda Branca (Padding)
                    # O EasyOCR falha se a letra estiver encostada na borda da imagem.
                    binary = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)

                    # Opcional: Visualizar o que o OCR está vendo (para debug)
                    # cv2.imshow("Debug OCR", binary) 

                    # 7. Leitura OCR na imagem binarizada
                    ocr_res = reader.readtext(binary, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')

                    
                    if ocr_res:
                        raw_text = ocr_res[0]
                        corrected_text = corrigir_placa(raw_text)
                        
                        if corrected_text:
                            # adiciona ao buffer do carro
                            if track_id not in memory_buffer:
                                memory_buffer[track_id] = []
                            
                            memory_buffer[track_id].append(corrected_text)
                            
                            # mantém apenas as últimas 15 leituras (janela deslizante)
                            if len(memory_buffer[track_id]) > 15:
                                memory_buffer[track_id].pop(0)

                # 3. votação com base na maioria das previsões
                final_text = ""
                confidence = 0.0
                if track_id in memory_buffer and len(memory_buffer[track_id]) > 0:
                    # pega o item mais comum na lista
                    most_common, count = Counter(memory_buffer[track_id]).most_common(1)[0]
                    final_text = most_common
                    confidence = count / len(memory_buffer[track_id]) # confianca. bom para debugar

                # 4. Desenha na tela
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if final_text:
                    cv2.putText(frame, final_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(frame, f"{confidence:.2f}", (x1, y1-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("LPR Estavel", frame)
    out.write(frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()