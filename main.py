import cv2
import easyocr
from ultralytics import YOLO
import re
from collections import Counter

# --- CONFIGURAÇÕES ---
video_path = 'videos/teste.mp4'
output_path = 'videos/resultado_estavel_blur.mp4'

# --- FUNÇÕES DE LIMPEZA E CORREÇÃO ---

def corrigir_placa(texto):
    """
    Força o padrão: 2 Letras, 2 Números, 3 Letras (Total 7 caracteres úteis)
    Ignora espaços ou traços na leitura bruta.
    """
    # 1. Limpeza: Remove tudo que não é letra ou número e deixa maiúsculo
    # Ex: "AB-12. cde" vira "AB12CDE"
    texto_limpo = re.sub(r'[^a-zA-Z0-9]', '', texto).upper()
    
    # O padrão deve ter exatamente 7 caracteres alfanuméricos
    if len(texto_limpo) != 7:
        return None

    # Dicionários de correção visual
    dict_char_to_num = {'O': '0', 'I': '1', 'J': '1', 'Z': '2', 'A': '4', 'S': '5', 'G': '6', 'B': '8', 'Q': '0', 'D': '0'}
    dict_num_to_char = {'0': 'O', '1': 'I', '2': 'Z', '4': 'A', '5': 'S', '6': 'G', '8': 'B'}

    lista_chars = list(texto_limpo)

    # --- APLICAÇÃO DA MÁSCARA (LLNNLLL) ---

    # Posições 0 e 1: TÊM que ser LETRAS
    for i in [0, 1]:
        if lista_chars[i] in dict_num_to_char:
            lista_chars[i] = dict_num_to_char[lista_chars[i]]

    # Posições 2 e 3: TÊM que ser NÚMEROS
    for i in [2, 3]:
        if lista_chars[i] in dict_char_to_num:
            lista_chars[i] = dict_char_to_num[lista_chars[i]]

    # Posições 4, 5 e 6: TÊM que ser LETRAS
    for i in [4, 5, 6]:
        if lista_chars[i] in dict_num_to_char:
            lista_chars[i] = dict_num_to_char[lista_chars[i]]

    # Reconstrói a string
    placa_final = "".join(lista_chars)

    # 2. Validação Final com Regex Rígido
    # ^ = inicio, [A-Z]{2} = 2 letras, [0-9]{2} = 2 numeros, [A-Z]{3} = 3 letras
    padrao_regex = re.compile(r'^[A-Z]{2}[0-9]{2}[A-Z]{3}$')
    
    if padrao_regex.match(placa_final):
        # Retorna formatado com espaço visual: "AB12 CDE"
        return f"{placa_final[:4]} {placa_final[4:]}"
    else:
        return None

# --- INICIALIZAÇÃO ---
print("Iniciando...")
model = YOLO('runs/detect/train4/weights/best.pt')  # Carrega o modelo treinado
reader = easyocr.Reader(['en'], gpu=True) # 'en' reconhece melhor letras/numeros gerais que 'pt'

cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# BUFFER DE MEMÓRIA (VOTAÇÃO)
# Estrutura: { track_id: [lista_de_leituras] }
memory_buffer = {}

while True:
    ret, frame = cap.read()
    if not ret: break

    # Use persist=True para manter o ID do carro
    results = model.track(frame, persist=True, verbose=False, tracker="bytetrack.yaml")

    for r in results:
        boxes = r.boxes
        for box in boxes:
            if box.id is not None:
                track_id = int(box.id.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Só processa se a caixa for grande o suficiente (placa muito longe dá erro)
                if (x2-x1) > 60 and (y2-y1) > 20:
                    
                    # 1. Recorte e Pré-processamento
                    crop = frame[y1:y2, x1:x2]
                    # Aumentar a imagem ajuda MUITO o OCR
                    crop_resized = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
                    # Aumentar contraste
                    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

                    # 2. Leitura OCR
                    # allowlist melhora precisão limitando caracteres possíveis
                    ocr_res = reader.readtext(gray, detail=0, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    
                    if ocr_res:
                        raw_text = ocr_res[0]
                        corrected_text = corrigir_placa(raw_text)
                        
                        if corrected_text:
                            # Adiciona ao buffer do carro
                            if track_id not in memory_buffer:
                                memory_buffer[track_id] = []
                            
                            memory_buffer[track_id].append(corrected_text)
                            
                            # Mantém apenas as últimas 15 leituras (janela deslizante)
                            if len(memory_buffer[track_id]) > 15:
                                memory_buffer[track_id].pop(0)

                # 3. Votação (A Mágica acontece aqui)
                final_text = ""
                if track_id in memory_buffer and len(memory_buffer[track_id]) > 0:
                    # Pega o item mais comum na lista
                    most_common, count = Counter(memory_buffer[track_id]).most_common(1)[0]
                    final_text = most_common
                    confidence = count / len(memory_buffer[track_id]) # Opcional: usar para debugar

                # 4. Desenha na tela
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if final_text:
                    cv2.putText(frame, final_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("LPR Estavel", frame)
    out.write(frame)
    if cv2.waitKey(1) == ord('q'): break

cap.release()
out.release()
cv2.destroyAllWindows()