import streamlit as st
import os
from dotenv import load_dotenv
import re
from typing import Dict, List, Optional
from dataclasses import dataclass
import openai
from openai import OpenAI

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(
    page_title="Prompt Optimizer - MIA Chatbot",
    page_icon="🔧",
    layout="wide"
)

# Tipos de prompts disponibles
PROMPT_TYPES = {
    "role_and_goal": "🎯 Rol y Objetivo",
    "tone_style_and_response_format": "💬 Tono, Estilo y Formato", 
    "conversation_flow": "🔄 Flujo Conversacional",
    "examples_interaction": "💡 Ejemplos de Interacción",
    "restrictions": "🚫 Restricciones"
}

@dataclass
class OptimizationResult:
    original_prompt: str
    optimized_prompt: str
    context_description: Optional[str]
    prompt_type: str
    has_context: bool
    optimization_mode: str
    context_analysis: str
    optimizations_applied: List[str]
    metaprompt_alignment: List[str]
    explanation: str
    best_practices_applied: List[str]
    compatibility_score: int
    relevance_analysis: Optional[str] # <-- MEJORA 2: Nuevo campo para el análisis de relevancia

class PromptOptimizer:
    def __init__(self):
        # Obtener API key del archivo .env
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY no encontrada en el archivo .env")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def optimize_prompt(self, user_prompt: str, prompt_type: str, context_description: str = "") -> OptimizationResult:
        """Optimiza un prompt usando la lógica del sistema."""
        
        has_context = bool(context_description.strip())
        
        optimization_prompt = self._create_optimization_template(
            user_prompt, context_description, prompt_type, has_context
        )
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": optimization_prompt}],
                temperature=0.1,
                max_tokens=2500
            )
            
            parsed_result = self._parse_response(response.choices[0].message.content, has_context)
            compatibility_score = self._calculate_mia_compatibility_score(
                user_prompt, parsed_result.get("optimized_prompt", ""), prompt_type
            )
            
            return OptimizationResult(
                original_prompt=user_prompt,
                optimized_prompt=parsed_result.get("optimized_prompt", ""),
                context_description=context_description if has_context else None,
                prompt_type=prompt_type,
                has_context=has_context,
                optimization_mode="contextualized" if has_context else "generic",
                context_analysis=parsed_result.get("context_analysis", ""),
                optimizations_applied=parsed_result.get("optimizations_applied", []),
                metaprompt_alignment=parsed_result.get("metaprompt_alignment", []),
                explanation=parsed_result.get("explanation", ""),
                best_practices_applied=parsed_result.get("best_practices_applied", []),
                compatibility_score=compatibility_score,
                relevance_analysis=parsed_result.get("relevance_analysis", "") # <-- MEJORA 2: Asignar resultado del análisis
            )
            
        except Exception as e:
            st.error(f"Error optimizando prompt: {str(e)}")
            return None
    
    def _create_optimization_template(self, user_prompt: str, context_description: str, 
                                    prompt_type: str, has_context: bool) -> str:
        """Template ajustado para mejora proactiva y estructural del prompt."""
        
        mia_system_context = """
SISTEMA MIA - CONTEXTO TÉCNICO:
MIA es un chatbot que debe poder:
- Consultar catálogos de productos/servicios cuando sea relevante
- Acceder a guías empresariales para información oficial  
- Detectar intenciones de compra automáticamente
- Escalar a agentes humanos cuando sea necesario
- Mantener conversaciones naturales y orientadas a resultados

RESTRICCIONES TÉCNICAS CRÍTICAS:
❌ No puede inventar información que no esté en bases de conocimiento o en el catálogo de productos
❌ No debe usar nombres técnicos de herramientas en respuestas al cliente
❌ No debe contradecir el flujo automático de escalamiento del sistema

TIPOS DE PROMPT A OPTIMIZAR:
1. 🎯 Rol y Objetivo: Define el rol del asistente y su objetivo principal.
2. 💬 Tono, Estilo y Formato: Especifica el tono de voz, estilo de respuesta y formato esperado.
3. 🔄 Flujo Conversacional: Describe cómo debe manejar el flujo de la conversación.
4. 💡 Ejemplos de Interacción: Proporciona ejemplos claros de cómo interactuar con los usuarios.
5. 🚫 Restricciones: Enumera las restricciones técnicas y de contenido que deben respetarse.
"""

        
        optimization_philosophy = """
FILOSOFÍA DE OPTIMIZACIÓN - ENRIQUECIMIENTO ESTRUCTURAL:
🎯 PRESERVAR: Mantén la personalidad, tono y objetivos clave del prompt original. La esencia del usuario y su intención en el prompt es la prioridad.
🔧 ESTRUCTURAR: Reorganiza y refina el prompt para que se comunique de la forma más clara y efectiva posible con el modelo de IA de MIA.
➕ ENRIQUECER: Agrega proactivamente las mejores prácticas de MIA si están ausentes. Esto incluye la consulta de herramientas, el manejo de información desconocida y el escalamiento adecuado, siempre adaptado al tono del usuario.
🚫 ELIMINAR: Solo elimina instrucciones que entren en conflicto directo con el funcionamiento de MIA.

PRINCIPIO CLAVE: "PRESERVA LA INTENCIÓN, PERO OPTIMIZA LA EJECUCIÓN"
- El objetivo no es solo corregir errores, sino elevar la calidad del prompt para maximizar el rendimiento de MIA.
- Si el prompt es bueno pero carece de estructura o de instrucciones clave (ej: qué hacer si no sabe algo), el optimizador debe añadirlas.
"""

        type_specific_guidance = {
            "role_and_goal": "...", # El contenido de las guías específicas se mantiene igual
            "tone_style_and_response_format": "...",
            "conversation_flow": "...",
            "examples_interaction": "...",
            "restrictions": "..."
        }
        # Para brevedad, se omite el contenido idéntico de type_specific_guidance
        specific_guidance = type_specific_guidance.get(prompt_type, "")


        context_section = ""
        if has_context and context_description.strip():
            context_section = f"""
CONTEXTO ESPECÍFICO DEL NEGOCIO:
"{context_description}"
"""
        else:
            context_section = "CONTEXTO DEL NEGOCIO: No se proporcionó contexto específico."

        return f"""
Eres un experto en optimización de prompts que mejora prompts preservando la intención del usuario y enriqueciéndolos con las mejores prácticas.

{mia_system_context}

{optimization_philosophy}

TIPO DE PROMPT A OPTIMIZAR: {prompt_type.upper().replace('_', ' ')}

{context_section}

PROMPT ORIGINAL DEL USUARIO:
"{user_prompt}"

METODOLOGÍA DE OPTIMIZACIÓN:
1. 🔍 ANALIZAR RELEVANCIA: Primero, determina si las instrucciones del usuario corresponden al tipo de prompt seleccionado.
2. 🎯 ANALIZAR CONTENIDO: Identifica la personalidad, objetivos y conflictos técnicos.
3. 🔧 MEJORAR Y ENRIQUECER: Re-escribe el prompt para que sea más claro, estructurado y completo, añadiendo las mejores prácticas de MIA si faltan, mientras preservas la esencia original. Elimina instrucciones que no correspondan al tipo de prompt.

FORMATO DE RESPUESTA OBLIGATORIO:

ANÁLISIS DE RELEVANCIA DEL CONTENIDO:
[Analiza si las instrucciones del prompt original corresponden al tipo de prompt seleccionado ('{PROMPT_TYPES[prompt_type]}'). Si detectas contenido que estaría mejor en otra sección (ej: detalles de tono en un prompt de 'Restricciones'), menciona cuáles fueron y a qué tipo de prompt deberían pertenecer. Si el contenido es relevante, confírmalo.]

ANÁLISIS DEL PROMPT ORIGINAL:
[Identificar elementos a preservar vs problemas técnicos a resolver y oportunidades de enriquecimiento.]

OPTIMIZACIONES APLICADAS:
[Lista específica de cambios y mejoras realizadas y su justificación técnica.]

PROMPT OPTIMIZADO:
[Versión mejorada (presentado en formato limpio) que PRESERVA la esencia original y resuelve conflictos técnicos y demás conflictos que pueda tener.]

MEJORAS IMPLEMENTADAS:
[Explicación de cómo se preservó el original mientras se mejoró la compatibilidad y efectividad, entre otros enriquecimientos.]

COMPATIBILIDAD CON MIA:
[Análisis (en formato simple de lista) de cómo el prompt optimizado se alinea con las mejores prácticas y restricciones del sistema MIA. Incluye una lista de verificación de compatibilidad, indicando si cumple con los requisitos técnicos y filosóficos del sistema.]
"""
    
    def _parse_response(self, response_content: str, has_context: bool) -> Dict:
        """Parsea la respuesta del modelo, extrayendo limpiamente todas las secciones."""
        try:
            sections = {}
            
            # <-- MEJORA 2: Nuevo patrón para extraer el análisis de relevancia
            relevance_pattern = r'ANÁLISIS DE RELEVANCIA DEL CONTENIDO:\s*(.*?)(?=\nANÁLISIS DEL PROMPT ORIGINAL:|$)'
            relevance_match = re.search(relevance_pattern, response_content, re.DOTALL)
            sections["relevance_analysis"] = relevance_match.group(1).strip() if relevance_match else ""

            # Análisis del prompt original
            analysis_pattern = r'ANÁLISIS DEL PROMPT ORIGINAL:\s*(.*?)(?=\nOPTIMIZACIONES APLICADAS:|$)'
            analysis_match = re.search(analysis_pattern, response_content, re.DOTALL)
            sections["context_analysis"] = analysis_match.group(1).strip() if analysis_match else ""
            
            # Optimizaciones aplicadas
            optimizations_match = re.search(r'OPTIMIZACIONES APLICADAS:\s*(.*?)(?=\nPROMPT OPTIMIZADO:|$)', response_content, re.DOTALL)
            optimizations_text = optimizations_match.group(1).strip() if optimizations_match else ""
            sections["optimizations_applied"] = self._parse_list_items(optimizations_text)
            
            # Prompt optimizado (LIMPIO)
            optimized_match = re.search(r'PROMPT OPTIMIZADO:\s*(.*?)(?=\nMEJORAS IMPLEMENTADAS:|$)', response_content, re.DOTALL)
            sections["optimized_prompt"] = optimized_match.group(1).strip() if optimized_match else "Error: No se pudo extraer el prompt optimizado"
            
            # Mejoras implementadas
            explanation_match = re.search(r'MEJORAS IMPLEMENTADAS:\s*(.*?)(?=\nCOMPATIBILIDAD CON MIA:|$)', response_content, re.DOTALL)
            sections["explanation"] = explanation_match.group(1).strip() if explanation_match else ""
            
            # Compatibilidad con MIA
            compatibility_match = re.search(r'COMPATIBILIDAD CON MIA:\s*(.*?)$', response_content, re.DOTALL)
            compatibility_text = compatibility_match.group(1).strip() if compatibility_match else ""
            sections["metaprompt_alignment"] = self._parse_list_items(compatibility_text)
            
            sections["best_practices_applied"] = self._extract_best_practices(sections["optimizations_applied"])
            
            return sections
            
        except Exception as e:
            return {
                "relevance_analysis": f"Error al procesar respuesta: {str(e)}",
                "context_analysis": f"Error al procesar respuesta: {str(e)}",
                "optimizations_applied": [],
                "optimized_prompt": "Error al extraer el prompt optimizado. Respuesta completa del modelo:\n\n" + response_content,
                "explanation": f"Error en el procesamiento: {str(e)}",
                "metaprompt_alignment": [],
                "best_practices_applied": []
            }
    
    def _parse_list_items(self, text: str) -> List[str]:
        """Convierte texto con bullets a lista."""
        items = []
        if text:
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('✅') or line.startswith('•') or line.startswith('-') or line.startswith('*')):
                    # Limpiar emojis y bullets
                    clean_line = re.sub(r'^[✅•\-*]\s*', '', line).strip()
                    if clean_line:
                        items.append(clean_line)
                elif line and not line.startswith('[') and len(line) > 15:
                    items.append(line)
        return items
    
    def _extract_best_practices(self, optimizations: List[str]) -> List[str]:
        """Extrae mejores prácticas de las optimizaciones aplicadas."""
        practices = []
        practice_keywords = {
            "preserv": "Preservación de elementos originales",
            "consulta": "Uso de consultas a fuentes oficiales",
            "escalamiento": "Escalamiento apropiado",
            "personalidad": "Mantenimiento de personalidad",
            "tono": "Preservación del tono original",
            "específico": "Personalización para el negocio",
            "compatibilidad": "Compatibilidad con sistema MIA"
        }
        
        for opt in optimizations:
            for keyword, practice in practice_keywords.items():
                if keyword in opt.lower() and practice not in practices:
                    practices.append(practice)
        
        return practices[:5]  # Máximo 5 prácticas
    
    def _calculate_mia_compatibility_score(self, original_prompt: str, optimized_prompt: str, prompt_type: str) -> int:
        """Sistema de validación específico para MIA con enfoque preservativo."""
        
        # Patrones críticos específicos de MIA que DEBEN estar
        critical_good_patterns = {
            "all_types": [
                "consulta", "información disponible", "fuentes", "conecta",
                "deriva", "equipo", "agente", "humano"
            ],
            "conversation_flow": [
                "si preguntan", "para información", "cuando no encuentres",
                "detecta", "facilita", "pasos"
            ],
            "role_and_goal": [
                "asistente", "ayuda", "consulta", "deriva cuando",
                "información", "disponible"
            ]
        }
        
        # Patrones que NUNCA deben aparecer
        critical_bad_patterns = [
            "inventa", "supón", "asume", "nunca digas que no sabes", 
            "siempre responde aunque no sepas", "retrieve_products", 
            "transfer_to_human"
        ]
        
        # NUEVO: Patrones de preservación (dan puntos por mantener elementos originales)
        preservation_patterns = [
            "amigable", "emojis", "😊", "🎉", "formal", "profesional",
            "casual", "divertido", "asistente"
        ]

        def count_patterns(text, patterns_list):
            text_lower = text.lower()
            return sum(1 for pattern in patterns_list if pattern in text_lower)

        def count_weighted_patterns(text, patterns_dict, prompt_type):
            text_lower = text.lower()
            score = 0
            
            # Patrones generales (peso 1)
            for pattern in patterns_dict.get("all_types", []):
                if pattern in text_lower:
                    score += 1
            
            # Patrones específicos del tipo (peso 2)
            for pattern in patterns_dict.get(prompt_type, []):
                if pattern in text_lower:
                    score += 2
                    
            return score

        # Calcular scores
        optimized_good = count_weighted_patterns(optimized_prompt, critical_good_patterns, prompt_type)
        optimized_bad = count_patterns(optimized_prompt, critical_bad_patterns)
        
        # NUEVO: Score de preservación (compara elementos originales mantenidos)
        original_elements = count_patterns(original_prompt, preservation_patterns)
        preserved_elements = count_patterns(optimized_prompt, preservation_patterns)
        preservation_score = min(preserved_elements, original_elements) if original_elements > 0 else 0
        
        # Calcular score final (0-10)
        base_score = min(optimized_good, 5)                    # Máximo 5 por patrones buenos
        preservation_bonus = min(preservation_score, 3)        # Máximo 3 por preservación
        penalty = min(optimized_bad * 3, 6)                   # Máximo 6 de penalización
        
        final_score = max(0, min(10, base_score + preservation_bonus - penalty + 2))  # +2 base
        
        return int(final_score)

def check_environment():
    """Verifica que el entorno esté configurado correctamente."""
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        st.error("**API Key no encontrada**")
        st.markdown("""
        **Necesitas configurar tu API Key de OpenAI en el archivo `.env`:**
        
        1. Crea un archivo `.env` en la misma carpeta que esta aplicación
        2. Agrega la línea: `OPENAI_API_KEY=tu_api_key_aquí`
        3. Reinicia la aplicación
        """)
        return False
    
    # Verificar que la API key tenga el formato correcto
    if not api_key.startswith('sk-'):
        st.warning("**Formato de API Key inválido**")
        st.markdown("La API Key de OpenAI debe comenzar con 'sk-'")
        return False
    
    return True

def main():
    st.title("Prompt Optimizer - MIA Chatbot")
    st.markdown("---")
    
    # Verificar configuración del entorno
    if not check_environment():
        st.stop()
    
    # Área principal - sin sidebar
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Entrada")
        
        # Formulario de entrada
        with st.form("prompt_optimizer_form"):
            # Prompt original
            user_prompt = st.text_area(
                "Prompt a Optimizar:",
                height=150,
                placeholder="Ejemplo: Eres Ana, súper amigable 😊 que usa emojis. Si no sabes algo, deriva al agente humano...",
                help="Escribe el prompt que quieres optimizar para el sistema MIA"
            )
            
            # Tipo de prompt
            prompt_type = st.selectbox(
                "Tipo de Prompt:",
                options=list(PROMPT_TYPES.keys()),
                format_func=lambda x: PROMPT_TYPES[x],
                help="Selecciona qué tipo de prompt estás optimizando"
            )
            
            # Contexto opcional
            st.markdown("**Contexto del Negocio (Opcional)**")
            context_description = st.text_area(
                "Describe tu negocio o contexto específico:",
                height=100,
                placeholder="Ejemplo: Tienda de ropa deportiva. Vendemos zapatillas Nike, Adidas. Atención presencial y online. Horarios 9-18h...",
                help="Esta información es opcional pero permite crear optimizaciones más específicas para tu negocio"
            )
            
            # Botón de optimización
            submitted = st.form_submit_button(
                "Optimizar Prompt",
                type="primary",
                use_container_width=True
            )
    
    with col2:
        st.header("Resultado")
        
        if submitted:
            if not user_prompt.strip():
                st.error("Por favor ingresa un prompt para optimizar")
            else:
                with st.spinner("Optimizando prompt..."):
                    try:
                        optimizer = PromptOptimizer()
                        result = optimizer.optimize_prompt(user_prompt, prompt_type, context_description)
                    except ValueError as e:
                        st.error(f"Error de configuración: {str(e)}")
                        result = None
                    except Exception as e:
                        st.error(f"Error inesperado: {str(e)}")
                        result = None
                
                if result:
                    # Mostrar resultados
                    display_results(result)


def display_results(result: OptimizationResult):
    """Muestra los resultados de la optimización con una pestaña dedicada para Relevancia."""
    
    # PROMPT OPTIMIZADO EN CAJA PRINCIPAL
    st.text_area(
        "Prompt Optimizado:",
        value=result.optimized_prompt,
        height=200,
        disabled=True,
        key="optimized_prompt_main"
    )
    
    st.code(result.optimized_prompt, language="text")
    
    # --- CAMBIO REALIZADO AQUÍ ---
    # Se añade una cuarta pestaña para "Relevancia".
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Análisis y Mejoras", 
        "🛠️ Optimizaciones Aplicadas", 
        "⚡ Compatibilidad", 
        "📌 Relevancia"
    ])
    
    with tab1:
        # Se elimina el análisis de relevancia de esta pestaña.
        st.markdown("**Análisis del Prompt Original**")
        if result.context_analysis:
            st.markdown(result.context_analysis)
        
        if result.explanation:
            st.markdown("---")
            st.markdown("**Mejoras Implementadas**")
            st.markdown(result.explanation)
    
    with tab2:
        st.markdown("**Lista de Optimizaciones Aplicadas**")
        if result.optimizations_applied:
            for i, opt in enumerate(result.optimizations_applied, 1):
                st.markdown(f"{i}. {opt}")
        else:
            st.info("No se aplicaron optimizaciones específicas, pero el prompt fue enriquecido según las mejores prácticas.")
    
    with tab3:
        st.markdown("**Alineación con el Sistema MIA**")
        if result.metaprompt_alignment:
            for alignment in result.metaprompt_alignment:
                st.markdown(f"✅ {alignment}")
        else:
            st.info("Compatible sin cambios adicionales necesarios.")

    # --- CAMBIO REALIZADO AQUÍ ---
    # Se crea el contenido para la nueva pestaña de Relevancia.
    with tab4:
        st.markdown("**Análisis de Relevancia del Contenido**")
        if result.relevance_analysis:
            st.markdown(result.relevance_analysis)
        else:
            st.info("No se generó un análisis de relevancia para este prompt.")
    
    # Comparación lado a lado al final
    st.markdown("---")
    st.subheader("Comparación")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Prompt Original:**")
        st.text_area(
            "Original",
            value=result.original_prompt,
            height=150,
            disabled=True,
            label_visibility="collapsed",
            key="original_comparison"
        )
    
    with col2:
        st.markdown("**Prompt Optimizado:**")
        st.text_area(
            "Optimizado",
            value=result.optimized_prompt,
            height=150,
            disabled=True,
            label_visibility="collapsed",
            key="optimized_comparison"
        )

if __name__ == "__main__":
    main()