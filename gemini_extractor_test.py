# -*- coding: utf-8 -*-
import os
import json
import re
import time
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image
import pytesseract
import sys

# --- Cargar Variables de Entorno (.env) ---
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Configuración de Tesseract (Opcional) ---
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Configurar la API de Gemini ---
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        print("API de Google AI configurada correctamente.")
    except Exception as e:
        print(f"Error al configurar la API de Google AI: {e}", file=sys.stderr)
        API_KEY = None
else:
    print("Advertencia: No se encontró GOOGLE_API_KEY en .env", file=sys.stderr)


def clean_ocr_text_for_llm(text: str) -> str:
    """Limpia texto OCR para LLM."""
    if not text: return ""
    text = re.sub(r'\n{2,}', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.replace('N? Transf.', 'N° Transf.')
    text = text.replace('N* de cuenta', 'N° de cuenta')
    text = text.replace('N* Operación', 'N° Operación')
    text = text.replace('N9 de cuenta', 'N° de cuenta')
    text = text.replace('N9 de usuario', 'N° de usuario')
    return text.strip()

def perform_ocr(image_path: str, ocr_config: str = '') -> str | None:
    """Realiza OCR."""
    try:
        custom_config = f'-l spa {ocr_config}'.strip()
        img = Image.open(image_path).convert('L')
        text = pytesseract.image_to_string(img, config=custom_config)
        print(f"--- OCR completado para: {os.path.basename(image_path)} (config: '{custom_config}') ---")
        return text
    except FileNotFoundError:
        print(f"Error: Imagen no encontrada: {image_path}", file=sys.stderr)
        return None
    except pytesseract.TesseractNotFoundError:
        print("\nError Crítico: 'tesseract' no encontrado.", file=sys.stderr)
        return "TESSERACT_ERROR"
    except Exception as e:
        print(f"Error OCR en {os.path.basename(image_path)}: {e}", file=sys.stderr)
        return None

def extract_data_with_gemini(ocr_text: str, filename: str) -> dict:
    """Usa Gemini para extraer datos (Prompt v6 - Fusión V4 lógica + V5 fixes)."""
    if not API_KEY: return {"error": "API Key no configurada."}
    if not ocr_text or not ocr_text.strip(): return {"error": "Texto OCR vacío."}

    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    cleaned_text = clean_ocr_text_for_llm(ocr_text)

    # ---- Prompt v6 - Fusión V4 + V5 ----
    prompt = f"""
    Analiza el siguiente texto OCR de un comprobante chileno ('{filename}') y extrae la info en JSON. Sigue ESTRICTAMENTE la estructura. Usa el valor JSON `null` si el dato no está CLARO. NO INVENTES.

    **Instrucciones Generales:**
    1.  **Tipo de Operación:** Primero determina si es:
        *   A) **Transferencia Normal:** Alguien ("Desde", "Origen") envía a otra persona/entidad ("Hacia", "Destino").
        *   B) **Pago a Comercio:** Alguien paga a un comercio usando una pasarela (PagoFácil, Webpay, Mercado Pago Pago QR). El destinatario es el comercio. Remitente suele ser `null`.
        *   C) **Retiro:** Alguien saca dinero de una plataforma (Mercado Pago) a su propia cuenta bancaria ("Retiraste dinero a...").
    2.  **Roles según Tipo:**
        *   Caso A: Remitente = Origen, Destinatario = Destino.
        *   Caso B: Remitente = `null` (a menos que se especifique), Destinatario = Nombre del Comercio.
        *   Caso C: Remitente = Plataforma Origen (ej: "Mercado Pago"), Destinatario = Datos de la cuenta bancaria receptora (Nombre titular si aparece, Banco, Cuenta).
    3.  **Banco Origen App:** Busca nombre/logo al inicio o título: "Mercado Pago", "BancoEstado App", "Pago Fácil", etc. Si es transferencia directa, usa el banco del remitente (si se conoce) o `null`.

    **Instrucciones Campos Específicos:**
    *   **Monto:** (Instrucción V5 exitosa) Busca "Monto", "Total". Encuentra el número. **IGNORA** puntos (`.`), comas (`,`) y ceros después de coma/punto (`,00`, `.00`). Extrae SÓLO el **número entero**. Ej: "$ 45.890,00" -> 45890. "$ 100.000" -> 100000.
    *   **RUT vs. Cuenta:** ¡Presta atención!
        *   Si tiene formato XXXXXXXX-X/K o XX.XXX.XXX-X/K -> `rut`.
        *   Si está etiquetado "Cuenta N°" o "N° de cuenta" -> `cuenta`.
        *   Si es número largo (>5 díg) sin formato RUT, bajo un nombre -> `cuenta` (Ej: '3216051001' bajo Maria Cerra).
        *   Si es número sin guion bajo "Usuario" (Ej: '343835988' bajo Jesus Iñaky) -> `rut`.
        *   Si está enmascarado ('*xXxx...') -> `rut`.
        *   Si es CuentaRUT (dice "CuentaRUT"), el `remitente.rut` puede ir también en `remitente.cuenta` si no hay otro número de cuenta explícito.
        *   En todos los demás casos, si no estás seguro, usa `null`. **NO PONGAS UN NÚMERO DE CUENTA EN EL CAMPO RUT NI VICEVERSA.**
    *   **Banco:** Asocia banco con remitente ("Desde", "Origen") o destinatario ("Hacia", "Destino", "Banco Destino").
    *   **Tipo Cuenta:** Busca CuentaRUT, Corriente, Vista, Ahorro. "Producto origen Ahorro" -> `remitente.tipo_cuenta`.
    *   **Fecha/Hora:** DD/MM/YYYY, YYYY-MM-DD, DD Mon YYYY, HH:MM:SS, HH:MM. Convierte "DD Dic YYYY" a "DD/12/YYYY".
    *   **Codigo Transaccion:** (Instrucción V5 exitosa) Busca "N° Transacción/Operación/Comprobante", "Folio", "ID", "Código Autorización", "Orden", "Pedido N°". **También busca números largos (>8 dígitos) aislados en la parte superior** (Ej: Mercado Pago `51787755999`).
    *   **Estado:** "Exitosa", "Realizado", "Aprobada", "Procesado", "Pagado", "Retirado", "Fallida".

    Texto del Comprobante (de {filename}):
    ```
    {cleaned_text}
    ```

    Estructura JSON Requerida (USA JSON `null`, NO el string `"null"`):
    {{
      "remitente": {{ "nombre": "string | null", "rut": "string | null", "banco": "string | null", "cuenta": "string | null", "tipo_cuenta": "string | null" }},
      "destinatario": {{ "nombre": "string | null", "rut": "string | null", "banco": "string | null", "cuenta": "string | null", "tipo_cuenta": "string | null" }},
      "monto": "number (entero) | null",
      "moneda": "string (CLP) | null", // Asume CLP si hay '$'
      "fecha": "string | null",
      "hora": "string | null",
      "asunto": "string | null", // Busca 'Asunto', 'Glosa', 'Descripción', 'Motivo'
      "codigo_transaccion": "string | null",
      "estado": "string | null",
      "banco_origen_app": "string | null"
    }}

    JSON extraído:
    """
    # ---- Fin Prompt v6 ----

    try:
        generation_config = genai.types.GenerationConfig(temperature=0.05) # Muy baja

        print(f"--- Llamando a la API de Gemini para {filename} ---")
        start_time = time.time()
        response = model.generate_content(prompt, generation_config=generation_config)
        end_time = time.time()
        print(f"--- Llamada completada en {end_time - start_time:.2f} segundos ---")

        # --- Manejo de Respuesta y Extracción/Parseo JSON (sin cambios) ---
        raw_response_text = None
        if hasattr(response, 'parts') and response.parts:
             raw_response_text = "".join(part.text for part in response.parts)
        elif response.candidates and response.candidates[0].finish_reason.name != "STOP":
             reason = response.candidates[0].finish_reason.name
             print(f"\nError API Gemini {filename}: Generación detenida.", file=sys.stderr)
             print(f"Razón: {reason}", file=sys.stderr)
             safety_ratings = ""
             try:
                  if response.candidates[0].safety_ratings: safety_ratings = f" Ratings: {response.candidates[0].safety_ratings}"
             except Exception: pass
             return {"error": f"Generación detenida: {reason}.{safety_ratings}"}
        elif hasattr(response, 'text'):
             raw_response_text = response.text
        else:
             print(f"\nError: Respuesta inesperada Gemini {filename} (sin 'text'/'parts').", file=sys.stderr)
             feedback = ""
             if hasattr(response, 'prompt_feedback'): feedback = f" Feedback: {response.prompt_feedback}"
             return {"error": f"Respuesta inválida Gemini.{feedback}", "raw_response": str(response)}

        if not raw_response_text:
            print(f"\nError: No se obtuvo texto de Gemini para {filename}.", file=sys.stderr)
            return {"error": "Respuesta Gemini sin texto.", "raw_response": str(response)}

        json_str = None
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw_response_text, re.DOTALL | re.IGNORECASE)
        if json_match:
            json_str = json_match.group(1)
        else:
            start = raw_response_text.find('{')
            end = raw_response_text.rfind('}')
            if start != -1 and end != -1 and start < end:
                 potential_json = raw_response_text[start:end+1]
                 if ':' in potential_json:
                     json_str = potential_json
                 else:
                      print(f"Advertencia: Texto {{}} no parece JSON {filename}.", file=sys.stderr)

        if not json_str:
             print(f"\nError: No se extrajo JSON de Gemini {filename}:\n{raw_response_text}", file=sys.stderr)
             return {"error": "No se extrajo JSON.", "raw_response": raw_response_text}

        try:
            json_str_cleaned = re.sub(r"//.*", "", json_str)
            json_str_cleaned = re.sub(r",\s*([\}\]])", r"\1", json_str_cleaned)
            # Reemplaza "null" string por null JSON value *antes* de parsear
            json_str_cleaned = json_str_cleaned.replace('"null"', 'null')
            extracted_data = json.loads(json_str_cleaned)
            # Post-validación/corrección opcional (ejemplo: asegurar monto entero)
            if 'monto' in extracted_data and isinstance(extracted_data['monto'], (float, str)):
                 try:
                     # Intenta convertir a float primero por si acaso, luego a int
                     extracted_data['monto'] = int(float(str(extracted_data['monto']).replace('.', '').replace(',', '')))
                 except (ValueError, TypeError):
                     extracted_data['monto'] = None # Poner null si la conversión falla
            return extracted_data
        except json.JSONDecodeError as json_e:
            print(f"\nError parseo JSON Gemini {filename}: {json_e}", file=sys.stderr)
            print(f"JSON intentado (limpio):\n{json_str_cleaned}", file=sys.stderr)
            return {"error": f"Error parseo JSON: {json_e}", "raw_response": raw_response_text, "json_attempted": json_str_cleaned}

    # --- Manejo de Excepciones API (sin cambios) ---
    except genai.types.generation_types.StopCandidateException as stop_e:
         print(f"\nError API Gemini {filename}: StopCandidateException.", file=sys.stderr)
         details = ""
         try:
             if stop_e.response.candidates and stop_e.response.candidates[0].finish_reason: details += f" Razón: {stop_e.response.candidates[0].finish_reason.name}."
             if stop_e.response.candidates and stop_e.response.candidates[0].safety_ratings: details += f" Ratings: {stop_e.response.candidates[0].safety_ratings}."
         except Exception: pass
         return {"error": f"Generación detenida.{details}"}
    except Exception as e:
        error_details = str(e)
        print(f"\nError llamada API Gemini {filename}: {error_details}", file=sys.stderr)
        if "API key not valid" in error_details: return {"error": "Error API: Clave inválida."}
        if "429" in error_details or "rate limit" in error_details.lower() or "resource has been exhausted" in error_details.lower(): return {"error": "Error API: Límite tasa/cuota."}
        if "Deadline exceeded" in error_details or "504" in error_details: return {"error": "Error API: Timeout."}
        if "safety settings" in error_details.lower() or "blocked" in error_details.lower(): return {"error": f"Error API: Bloqueo seguridad. {error_details}"}
        return {"error": f"Error inesperado API Gemini: {error_details}"}


# --- Bloque Principal ---
if __name__ == "__main__":
    print("Iniciando procesador de comprobantes v6...") # v6!

    if not API_KEY:
        print("\nERROR CRÍTICO: GOOGLE_API_KEY inválida o no encontrada.", file=sys.stderr)
        exit(1)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    boletas_dir = os.path.join(base_dir, "boletas")

    if not os.path.isdir(boletas_dir):
        print(f"\nERROR CRÍTICO: Carpeta '{boletas_dir}' no encontrada.", file=sys.stderr)
        exit(1)

    supported_extensions = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')
    try:
        all_files = os.listdir(boletas_dir)
        image_files = sorted([f for f in all_files
                       if os.path.isfile(os.path.join(boletas_dir, f)) and f.lower().endswith(supported_extensions)])
    except OSError as e:
        print(f"\nERROR CRÍTICO: No se pudo leer carpeta '{boletas_dir}': {e}", file=sys.stderr)
        exit(1)

    if not image_files:
        print(f"\nAdvertencia: No se encontraron imágenes en '{boletas_dir}'.")
        exit(0)

    print(f"\nSe encontraron {len(image_files)} imágenes. Procesando...")
    print("-" * 50)

    tesseract_ok = True
    ocr_processing_config = '' # Mantenemos default OCR
    results = {}

    for filename in image_files:
        print(f"\n--- Procesando Archivo: {filename} ---")
        image_path = os.path.join(boletas_dir, filename)

        if not tesseract_ok:
            print("Saltando OCR por error previo.", file=sys.stderr)
            results[filename] = {"error": "Tesseract no disponible."}
            continue

        ocr_result = perform_ocr(image_path, ocr_config=ocr_processing_config)

        if ocr_result == "TESSERACT_ERROR":
            tesseract_ok = False
            print("\nError fatal Tesseract. Deteniendo.", file=sys.stderr)
            results[filename] = {"error": "Tesseract no funcional."}
            break
        elif ocr_result is None:
            print(f"Fallo OCR {filename}. Saltando.")
            results[filename] = {"error": "Fallo OCR."}
            print("-" * 50)
            continue
        elif not ocr_result.strip():
            print(f"Advertencia: OCR vacío {filename}. Saltando Gemini.")
            results[filename] = {"error": "OCR vacío."}
            print("-" * 50)
            continue

        print("\n--- Texto OCR Crudo (Antes de limpiar) ---")
        print(ocr_result)
        print("--- Fin Texto OCR Crudo ---")

        extracted_data = extract_data_with_gemini(ocr_result, filename)
        results[filename] = extracted_data

        print(f"\n--- Resultado Extracción para {filename} ---")
        if isinstance(extracted_data, dict) and "error" in extracted_data:
            print(f"Error: {extracted_data['error']}", file=sys.stderr)
            if "raw_response" in extracted_data: print("Raw Response:", extracted_data["raw_response"], file=sys.stderr)
            if "json_attempted" in extracted_data: print("JSON Intentado:", extracted_data["json_attempted"], file=sys.stderr)
        elif isinstance(extracted_data, dict):
             print(json.dumps(extracted_data, indent=2, ensure_ascii=False))
        else:
             print(f"Error: Resultado inesperado: {extracted_data}", file=sys.stderr)

        print("-" * 50)

    # --- Resumen Final (sin cambios) ---
    print("\n--- Resumen del Procesamiento ---")
    successful_extractions = 0
    ocr_failures = 0
    gemini_failures = 0
    for fname, data in results.items():
        if isinstance(data, dict):
            if "error" not in data: successful_extractions += 1
            elif "OCR" in data.get("error", "") or "Tesseract" in data.get("error", ""): ocr_failures += 1
            else: gemini_failures += 1
        else: gemini_failures += 1

    print(f"Total imágenes procesadas: {len(results)} / {len(image_files)}")
    print(f"Extracciones exitosas (JSON): {successful_extractions}")
    print(f"Fallos OCR/Tesseract: {ocr_failures}")
    print(f"Fallos API Gemini/JSON: {gemini_failures}")
    print("--- Procesamiento completado ---")