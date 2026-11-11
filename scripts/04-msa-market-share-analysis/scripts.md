4. SCRIPTS MSA (MARKET SHARE ANALYSIS)
Nota: Los siguientes scripts están diseñados para el análisis de datos orientado a la estimación de market share, la generación de visualizaciones analíticas y la identificación de insights estratégicos a partir de datos históricos obtenidos de TradeMap. Se sugiere implementar este flujo de trabajo en entornos de ejecución Python tales como:
Google Colab
Python-Fiddle
Kaggle Notebooks
Deepnote
4.1. Importar e instalar librerías
```
# Instalar librerías si es necesario
!pip install openpyxl

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo de gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
```
4.2. Cargar archivos fuente (Trademap)
Nota: El script identifica automáticamente la columna de países y las columnas de valores por año, aunque las cabeceras estén en español o en inglés o tengan nombres distintos. Si la detección no es concluyente, muestra las columnas disponibles (nombre e índice) y pide seleccionarlas manualmente. También se pueden asignar desde el inicio en la sección "CONFIGURACIÓN OPCIONAL". Admite archivos .xlsx y .xls.
```
# Soporte para archivos .xls (además de .xlsx)
!pip install -q openpyxl xlrd

import re
import unicodedata

# ============================================================
# CONFIGURACIÓN OPCIONAL: ASIGNACIÓN MANUAL DE COLUMNAS
# Dejar en None para usar la detección automática.
# Se pueden indicar nombres o índices (posiciones) de columna.
#   Ejemplos:
#   COLUMNA_PAIS_MANUAL   = 'Exportadores'      o  0
#   COLUMNAS_VALOR_MANUAL = ['Valor importado en 2020', 'Valor importado en 2021']
#                           o  [1, 2, 3, 4, 5]   o  '1-5'
# ============================================================
COLUMNA_PAIS_MANUAL = None
COLUMNAS_VALOR_MANUAL = None

# Palabras clave reconocidas (español e inglés)
KEYWORDS_PAIS = ['exportador', 'importador', 'exporter', 'importer', 'socio', 'partner',
                 'pais', 'country', 'countries', 'mercado', 'market', 'proveedor',
                 'supplier', 'origen', 'origin', 'destino', 'destination']
KEYWORDS_VALOR = ['valor', 'value', 'importad', 'exportad', 'imported', 'exported',
                  'import', 'export', 'comercio', 'trade', 'usd', 'dolar', 'dollar']
KEYWORDS_EXCLUIR = ['cantidad', 'quantity', 'unidad', 'unit', 'tasa', 'rate', 'growth',
                    'crecimiento', 'participacion', 'share', '%', 'distancia', 'distance',
                    'balanza', 'balance', 'arancel', 'tariff', 'concentracion', 'concentration']
FILAS_TOTALES = ['mundo', 'world', 'total', 'total general', 'grand total']


# ---------- Funciones auxiliares ----------
def normalizar(texto):
    """Minúsculas, sin tildes y sin espacios extra."""
    texto = unicodedata.normalize('NFKD', str(texto))
    texto = ''.join(c for c in texto if not unicodedata.combining(c))
    return ' '.join(texto.lower().split())

def extraer_anio(texto):
    """Devuelve el año (1950-2100) contenido en una cabecera, o None."""
    anios = [int(a) for a in re.findall(r'(?<!\d)(\d{4})(?!\d)', str(texto)) if 1950 <= int(a) <= 2100]
    return anios[-1] if anios else None

def a_numero(serie):
    """Convierte a número valores como '1,234' o ' 567 '."""
    if pd.api.types.is_numeric_dtype(serie):
        return serie
    limpia = (serie.astype(str)
                   .str.replace('\xa0', '', regex=False)
                   .str.replace(' ', '', regex=False)
                   .str.replace(',', '', regex=False))
    return pd.to_numeric(limpia, errors='coerce')

def proporcion_numerica(serie):
    datos = serie.dropna()
    return a_numero(datos).notna().mean() if len(datos) else 0.0

def leer_archivo_sin_cabecera(nombre):
    """Lee .xlsx/.xls. Algunos .xls de Trade Map son en realidad tablas HTML o texto."""
    def leer_html():
        tabla = pd.read_html(nombre, encoding='utf-8')[0]
        cab = [' '.join(str(x) for x in c if 'Unnamed' not in str(x)) if isinstance(c, tuple) else c
               for c in tabla.columns]
        return pd.DataFrame([cab] + tabla.values.tolist())   # conservar la cabecera como fila
    errores = []
    for lector in (lambda: pd.read_excel(nombre, sheet_name=0, header=None),
                   leer_html,
                   lambda: pd.read_csv(nombre, sep='\t', header=None, encoding='utf-8'),
                   lambda: pd.read_csv(nombre, sep='\t', header=None, encoding='latin-1')):
        try:
            tabla = lector()
            if tabla.shape[1] > 1:
                return tabla
        except Exception as e:
            errores.append(str(e))
    raise ValueError("❌ No fue posible leer el archivo. Verifica que sea un Excel (.xlsx o .xls) "
                     f"exportado desde Trade Map.\nDetalles: {errores}")

def detectar_fila_cabecera(tabla, max_filas=25):
    """Busca la fila con más cabeceras de años y palabras clave de país."""
    mejor_fila, mejor_puntaje = 0, -1
    for i in range(min(max_filas, len(tabla))):
        celdas = [normalizar(v) for v in tabla.iloc[i].values if pd.notna(v)]
        n_anios = sum(1 for c in celdas if extraer_anio(c))
        n_pais = sum(1 for c in celdas if any(k in c for k in KEYWORDS_PAIS))
        puntaje = n_anios * 2 + n_pais
        if n_anios >= 1 and puntaje > mejor_puntaje:
            mejor_fila, mejor_puntaje = i, puntaje
    return mejor_fila

def mostrar_columnas(df_cols):
    """Interfaz de selección: muestra índice, nombre, ejemplo y % numérico."""
    print("\n📋 COLUMNAS DISPONIBLES EN EL ARCHIVO")
    print("-" * 80)
    print(f"{'Índice':<7} {'Nombre de la columna':<45} {'Ejemplo':<15} {'% numérico':>10}")
    print("-" * 80)
    for i, col in enumerate(df_cols.columns):
        no_nulos = df_cols[col].dropna()
        ejemplo = str(no_nulos.iloc[0])[:14] if len(no_nulos) else '(vacía)'
        print(f"{i:<7} {str(col)[:44]:<45} {ejemplo:<15} {proporcion_numerica(df_cols[col]):>9.0%}")
    print("-" * 80)

def resolver_columnas(entrada, columnas):
    """Convierte nombres, índices o rangos ('1-5', '1,2,3') en nombres de columnas."""
    if entrada is None:
        return []
    if isinstance(entrada, (int, str)):
        entrada = [entrada]
    elementos = []
    for e in entrada:
        if isinstance(e, str) and e.strip() not in columnas:
            for parte in e.split(','):
                parte = parte.strip()
                if re.fullmatch(r'\d+\s*-\s*\d+', parte):
                    ini, fin = [int(x) for x in parte.split('-')]
                    elementos.extend(range(ini, fin + 1))
                elif parte.isdigit():
                    elementos.append(int(parte))
                elif parte:
                    elementos.append(parte)
        else:
            elementos.append(e.strip() if isinstance(e, str) else e)
    resultado = []
    for e in elementos:
        if isinstance(e, int):
            if not 0 <= e < len(columnas):
                raise ValueError(f"❌ El índice {e} no existe. Índices válidos: 0 a {len(columnas)-1}.")
            resultado.append(columnas[e])
        elif e in columnas:
            resultado.append(e)
        else:
            raise ValueError(f"❌ La columna '{e}' no existe en el archivo.")
    return list(dict.fromkeys(resultado))

def detectar_columna_pais(tabla, excluidas):
    """Devuelve (columna, es_concluyente)."""
    candidatas = [c for c in tabla.columns if c not in excluidas]
    por_nombre = [c for c in candidatas if any(k in normalizar(c) for k in KEYWORDS_PAIS)
                  and proporcion_numerica(tabla[c]) < 0.5]
    if len(por_nombre) == 1:
        return por_nombre[0], True
    if len(por_nombre) > 1:
        return por_nombre[0], False
    # Por características: columnas mayoritariamente de texto
    texto = [c for c in candidatas if tabla[c].notna().mean() > 0.5 and proporcion_numerica(tabla[c]) < 0.2]
    if len(texto) == 1:
        return texto[0], True
    return (texto[0] if texto else None), False

def detectar_columnas_valor(tabla):
    """Devuelve (lista de columnas, es_concluyente)."""
    con_anio = [c for c in tabla.columns if extraer_anio(c)
                and not any(k in normalizar(c) for k in KEYWORDS_EXCLUIR)
                and proporcion_numerica(tabla[c]) >= 0.5]
    con_clave = [c for c in con_anio if any(k in normalizar(c) for k in KEYWORDS_VALOR)]
    elegidas = con_clave if con_clave else con_anio
    anios = [extraer_anio(c) for c in elegidas]
    concluyente = len(elegidas) > 0 and len(anios) == len(set(anios))
    return elegidas, concluyente

def validar_columnas(tabla, col_pais, cols_valor):
    """Verifica que las columnas sean compatibles con el procesamiento."""
    problemas = []
    if col_pais is None:
        problemas.append("No se identificó la columna de países.")
    elif proporcion_numerica(tabla[col_pais]) > 0.5:
        problemas.append(f"La columna de países '{col_pais}' contiene principalmente números.")
    if not cols_valor:
        problemas.append("No se identificaron columnas de valores comerciales.")
    if col_pais in cols_valor:
        problemas.append("La columna de países no puede ser también una columna de valores.")
    sin_anio = [c for c in cols_valor if extraer_anio(c) is None]
    if sin_anio:
        problemas.append(f"Estas columnas no indican un año en su cabecera: {sin_anio}")
    anios = [extraer_anio(c) for c in cols_valor if extraer_anio(c)]
    repetidos = sorted({a for a in anios if anios.count(a) > 1})
    if repetidos:
        problemas.append(f"Hay años repetidos en las columnas de valores: {repetidos}")
    no_numericas = [c for c in cols_valor if c != col_pais and proporcion_numerica(tabla[c]) < 0.5]
    if no_numericas:
        problemas.append(f"Estas columnas no contienen valores numéricos: {no_numericas}")
    return problemas

def seleccion_manual(tabla, col_pais_sugerida, cols_valor_sugeridas):
    """Pide al usuario elegir las columnas por nombre o índice."""
    mostrar_columnas(tabla)
    columnas = list(tabla.columns)
    print("\n✍️  ASIGNACIÓN MANUAL (Enter = aceptar la sugerencia)")
    try:
        sug_p = columnas.index(col_pais_sugerida) if col_pais_sugerida in columnas else ''
        r = input(f"Índice o nombre de la columna de PAÍSES [{sug_p}]: ").strip()
        col_pais = resolver_columnas(r, columnas)[0] if r else col_pais_sugerida
        sug_v = ','.join(str(columnas.index(c)) for c in cols_valor_sugeridas)
        r = input(f"Índices o nombres de las columnas de VALORES por año (ej. 1-5 o 1,2,3) [{sug_v}]: ").strip()
        cols_valor = resolver_columnas(r, columnas) if r else cols_valor_sugeridas
    except EOFError:
        raise ValueError("❌ No se pudo pedir la selección manual. Asigna las columnas en "
                         "COLUMNA_PAIS_MANUAL y COLUMNAS_VALOR_MANUAL y vuelve a ejecutar.")
    return col_pais, cols_valor


# ---------- 1. Cargar archivo ----------
# Subir archivo Excel directamente desde TradeMap
uploaded = files.upload()
filename = list(uploaded.keys())[0]

tabla_completa = leer_archivo_sin_cabecera(filename)
fila_cabecera = detectar_fila_cabecera(tabla_completa)

# Leer datos tal como vienen de TradeMap, usando la fila de cabecera detectada
df_raw = tabla_completa.iloc[fila_cabecera + 1:].copy()
cabeceras = [str(c).strip() if pd.notna(c) else f'Columna_{i}'
             for i, c in enumerate(tabla_completa.iloc[fila_cabecera].values)]
vistos = {}
for i, c in enumerate(cabeceras):          # evitar nombres de columna repetidos
    if c in vistos:
        vistos[c] += 1
        cabeceras[i] = f"{c}.{vistos[c]}"
    else:
        vistos[c] = 0
df_raw.columns = cabeceras
df_raw = df_raw.dropna(how='all').dropna(axis=1, how='all').reset_index(drop=True)

# Verificar estructura original
print(f"Fila de cabecera detectada: {fila_cabecera + 1}")
print("Columnas disponibles:", df_raw.columns.tolist())
print("\nPrimeras 5 filas:")
print(df_raw.head())

# ---------- 2. Identificar columnas ----------
columnas = list(df_raw.columns)
if COLUMNA_PAIS_MANUAL is not None or COLUMNAS_VALOR_MANUAL is not None:
    print("\n🔧 Usando asignación manual definida en la configuración.")
    auto_valor, _ = detectar_columnas_valor(df_raw)
    year_columns = resolver_columnas(COLUMNAS_VALOR_MANUAL, columnas) or auto_valor
    auto_pais, _ = detectar_columna_pais(df_raw, year_columns)
    country_col = resolver_columnas(COLUMNA_PAIS_MANUAL, columnas)[0] if COLUMNA_PAIS_MANUAL is not None else auto_pais
else:
    year_columns, valor_ok = detectar_columnas_valor(df_raw)
    country_col, pais_ok = detectar_columna_pais(df_raw, year_columns)
    if valor_ok and pais_ok:
        print("\n✅ Detección automática concluyente.")
    else:
        print("\n⚠️  La detección automática no fue concluyente.")
        country_col, year_columns = seleccion_manual(df_raw, country_col, year_columns)

# ---------- 3. Validar correspondencias ----------
problemas = validar_columnas(df_raw, country_col, year_columns)
if problemas:
    mostrar_columnas(df_raw)
    raise ValueError("❌ Inconsistencias encontradas:\n  - " + "\n  - ".join(problemas) +
                     "\nAjusta COLUMNA_PAIS_MANUAL / COLUMNAS_VALOR_MANUAL y vuelve a ejecutar.")

print(f"Columna de países: '{country_col}'")
print(f"Columnas de años encontradas: {year_columns}")
print("Correspondencia año → columna:")
for col in year_columns:
    print(f"  {extraer_anio(col)} → {col}")

# ---------- 4. Transformar de formato ancho a formato largo (melt) ----------
df_melted = df_raw.melt(
    id_vars=[country_col],
    value_vars=year_columns,
    var_name='year_column',
    value_name='trade_value_usd'
)

# Extraer año de la columna
df_melted['year'] = df_melted['year_column'].apply(extraer_anio).astype(int)
df_melted = df_melted.rename(columns={country_col: 'partner_country'})

# Limpiar datos
df_melted['partner_country'] = df_melted['partner_country'].astype(str).str.strip()
df_melted['trade_value_usd'] = a_numero(df_melted['trade_value_usd'])
df = df_melted[['year', 'partner_country', 'trade_value_usd']].copy()

# Eliminar filas con datos faltantes y fila "Mundo" / "World"
df = df.dropna(subset=['trade_value_usd'])
df = df[~df['partner_country'].apply(normalizar).isin(FILAS_TOTALES + ['nan', ''])].reset_index(drop=True)

if df.empty:
    raise ValueError("❌ Después de la limpieza no quedaron datos. Revisa las columnas seleccionadas.")

print(f"\nDatos procesados: {len(df)} filas")
print(f"Países únicos: {df['partner_country'].nunique()}")
print(f"Años disponibles: {sorted(df['year'].unique())}")
print("\nPrimeros datos procesados:")
print(df.head(10))
```
4.3. Calcular Market Share, Ranking y aplicar filtros
```
# Calcular market share por año y país
market_evolution = df.groupby(['year', 'partner_country'])['trade_value_usd'].sum().reset_index()

# Calcular market share total por año
yearly_totals = market_evolution.groupby('year')['trade_value_usd'].sum()
market_evolution['market_share'] = market_evolution.apply(
    lambda x: (x['trade_value_usd'] / yearly_totals[x['year']]) * 100, axis=1
)

# Identificar top países
top_countries = df.groupby('partner_country')['trade_value_usd'].sum().nlargest(8).index

# Filtrar solo top países
market_evolution_top = market_evolution[market_evolution['partner_country'].isin(top_countries)]
```
4.4.  Generar Tablas Pivot para análisis
```
# TABLA 1: Market Share por año y país
print("📊 TABLA 1: MARKET SHARE POR AÑO (%)")
print("="*60)
pivot_share = market_evolution_top.pivot(index='year', columns='partner_country', values='market_share')
pivot_share_rounded = pivot_share.round(1)
print(pivot_share_rounded.to_string())

# TABLA 2: Valores de comercio (Millones USD)
print("\n\n💰 TABLA 2: VALORES DE COMERCIO (Millones USD)")
print("="*60)
pivot_value = market_evolution_top.pivot(index='year', columns='partner_country', values='trade_value_usd')
pivot_value_millions = (pivot_value / 1000).round(1)  # Convertir a millones
print(pivot_value_millions.to_string())

# TABLA 3: Crecimiento anual por país
print("\n\n📈 TABLA 3: CRECIMIENTO ANUAL COMPUESTO (CAGR %)")
print("="*60)
growth_data = market_evolution_top.groupby('partner_country').apply(
    lambda x: ((x['trade_value_usd'].iloc[-1] / x['trade_value_usd'].iloc[0]) ** (1/(len(x)-1)) - 1) * 100
).reset_index(name='cagr')
growth_summary = growth_data.set_index('partner_country')['cagr'].round(1).sort_values(ascending=False)
print(growth_summary.to_string())

# TABLA 4: Ranking por año
print("\n\n🏆 TABLA 4: RANKING DE PAÍSES POR AÑO (Top 5)")
print("="*60)
for year in sorted(df['year'].unique()):
    year_data = market_evolution_top[market_evolution_top['year'] == year].nlargest(5, 'market_share')
    print(f"\n{year}:")
    for i, (_, row) in enumerate(year_data.iterrows(), 1):
        print(f"  {i}. {row['partner_country']}: {row['market_share']:.1f}% ({row['trade_value_usd']/1000:.0f}M USD)")
```
4.5. Generar Gráficos analíticos
Nota: Al ejecutar este bloque aparece un menú para elegir la paleta de colores de los gráficos: tres paletas predeterminadas o una paleta personalizada. Para la personalizada, pegar el código CSS exportado desde Coolors (Export → Code → CSS) y escribir `FIN` en una línea nueva (o presionar Enter en una línea vacía). El CSS no se ejecuta: solo se extraen y validan los códigos de color HEX. No es necesario modificar el código.
```
import re
from matplotlib.colors import LinearSegmentedColormap, to_rgb, to_hex, is_color_like

# ============================================================
# DEFINICIÓN CENTRALIZADA DE PALETAS
# ============================================================
PALETAS_PREDETERMINADAS = {
    '1': {
        'nombre': 'Ejecutivo Clásico',
        'descripcion': 'Azules sobrios, grises y acentos dorados para informes ejecutivos',
        'colores': ['#1F3A5F', '#3E6D9C', '#6C8EBF', '#5B6770', '#A3ADB8', '#C9A227', '#D9822B', '#9DB4D3'],
    },
    '2': {
        'nombre': 'Comercio Global',
        'descripcion': 'Azules y verdes que evocan mercados internacionales y competitividad',
        'colores': ['#0B3C5D', '#1D6FA3', '#0F9D8A', '#2E8B57', '#7BC67E', '#328CC1', '#F2B134', '#ED553B'],
    },
    '3': {
        'nombre': 'Corporativo Contemporáneo',
        'descripcion': 'Neutros, tierras y acentos cálidos para reportes analíticos modernos',
        'colores': ['#3D3B38', '#8C7B6B', '#B5651D', '#D4A373', '#6B705C', '#A3B18A', '#E07A5F', '#F2CC8F'],
    },
}
OPCION_PERSONALIZADA = '4'
MIN_COLORES_PERSONALIZADOS = 2


# ============================================================
# FUNCIONES DE PALETA
# ============================================================
def luminancia(color):
    r, g, b = to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b

def oscurecer(color, factor=0.45):
    r, g, b = to_rgb(color)
    return to_hex((r * factor, g * factor, b * factor))

def extraer_colores_css(texto_css):
    """Extrae los colores HEX de un texto CSS SIN ejecutarlo.
    Admite #RGB, #RRGGBB y #RRGGBBAA (formato de Coolors), incluso dentro de :root { }."""
    encontrados = re.findall(r'#([0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3})(?![0-9a-fA-F])', str(texto_css))
    colores = []
    for h in encontrados:
        if len(h) == 3:
            h = ''.join(c * 2 for c in h)
        h = '#' + h[:6].upper()          # se ignora el canal alfa
        if is_color_like(h) and h not in colores:
            colores.append(h)
    return colores

def validar_paleta(colores):
    """Devuelve una lista de problemas (vacía si la paleta es utilizable)."""
    problemas = []
    if len(colores) < MIN_COLORES_PERSONALIZADOS:
        problemas.append(f"Se encontraron {len(colores)} color(es) HEX válido(s); se necesitan al menos "
                         f"{MIN_COLORES_PERSONALIZADOS}. Ejemplo de formato: --azul: #1F3A5F;")
    elif len([c for c in colores if luminancia(c) < 0.85]) == 0:
        problemas.append("Todos los colores son demasiado claros y no serían legibles sobre fondo blanco.")
    return problemas

def construir_esquema(nombre, colores, n_series):
    """Asigna roles a los colores de la paleta (series, texto, cuadrícula, etc.)."""
    utilizables = [c for c in colores if luminancia(c) < 0.85] or colores
    claros = [c for c in colores if luminancia(c) >= 0.85]
    if len(utilizables) >= n_series:
        series = utilizables[:n_series]
    else:  # generar tonos intermedios a partir de la misma paleta
        cmap = LinearSegmentedColormap.from_list('paleta', utilizables if len(utilizables) > 1 else utilizables * 2)
        series = [to_hex(cmap(i / max(n_series - 1, 1))) for i in range(n_series)]
    oscuro = min(colores, key=luminancia)
    return {
        'nombre': nombre,
        'colores': colores,
        'series': series,
        'texto': oscuro if luminancia(oscuro) < 0.3 else oscurecer(oscuro),
        'cuadricula': claros[0] if claros else to_hex([0.5 + 0.5 * v for v in to_rgb(utilizables[-1])]),
        'positivo': utilizables[0],
        'negativo': utilizables[-1] if len(utilizables) > 1 else oscurecer(utilizables[0], 0.7),
    }

def mostrar_muestras(paletas):
    """Vista previa de las paletas disponibles."""
    fig_p, axs = plt.subplots(len(paletas), 1, figsize=(10, 1.1 * len(paletas)))
    axs = np.atleast_1d(axs)
    for ax, (clave, p) in zip(axs, paletas.items()):
        for i, c in enumerate(p['colores']):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=c))
            ax.text(i + 0.5, -0.25, c, ha='center', va='top', fontsize=7)
        ax.set_xlim(0, len(p['colores'])); ax.set_ylim(-0.6, 1); ax.axis('off')
        ax.set_title(f"{clave}. {p['nombre']}", loc='left', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.show()

def leer_css():
    """Campo de texto para pegar el CSS (varias líneas). Termina con FIN o una línea vacía."""
    print("\n📋 Pega el código CSS de tu paleta (por ejemplo, exportado desde Coolors).")
    print("   Cuando termines, escribe FIN en una línea nueva o presiona Enter en una línea vacía.")
    lineas = []
    while True:
        linea = input()
        if linea.strip().upper() == 'FIN' or (linea.strip() == '' and lineas):
            break
        lineas.append(linea)
    return '\n'.join(lineas)

def seleccionar_paleta(max_intentos=3):
    """Interfaz de selección: devuelve (nombre, lista_de_colores)."""
    print("🎨 SELECCIÓN DE PALETA DE COLORES PARA LOS GRÁFICOS")
    print("=" * 60)
    for clave, p in PALETAS_PREDETERMINADAS.items():
        print(f"  {clave}. {p['nombre']:<27} {p['descripcion']}")
    print(f"  {OPCION_PERSONALIZADA}. {'Paleta personalizada (CSS)':<27} Pegar el código CSS de una paleta propia")
    mostrar_muestras(PALETAS_PREDETERMINADAS)
    try:
        for _ in range(max_intentos):
            opcion = input(f"Escribe el número de la paleta (1-{OPCION_PERSONALIZADA}) y presiona Enter: ").strip()
            if opcion in PALETAS_PREDETERMINADAS:
                p = PALETAS_PREDETERMINADAS[opcion]
                return p['nombre'], p['colores']
            if opcion == OPCION_PERSONALIZADA:
                for _ in range(max_intentos):
                    colores = extraer_colores_css(leer_css())
                    problemas = validar_paleta(colores)
                    if not problemas:
                        print(f"✅ Se encontraron {len(colores)} colores válidos: {', '.join(colores)}")
                        return 'Paleta personalizada', colores
                    print("❌ El código pegado no permite construir una paleta:")
                    for pr in problemas:
                        print(f"   - {pr}")
                    print("   Inténtalo de nuevo.")
                print("⚠️  No se recibió un CSS válido. Elige otra opción.")
                continue
            print(f"⚠️  Opción no válida: '{opcion}'. Escribe un número del 1 al {OPCION_PERSONALIZADA}.")
    except EOFError:
        print("⚠️  No fue posible leer la selección (entorno no interactivo).")
    p = PALETAS_PREDETERMINADAS['1']
    print(f"ℹ️  Se usará la paleta predeterminada: {p['nombre']}.")
    return p['nombre'], p['colores']


# ============================================================
# SELECCIÓN Y APLICACIÓN DE LA PALETA
# ============================================================
nombre_paleta, colores_paleta = seleccionar_paleta()
paises_grafico = sorted(set(pivot_share.columns) | set(growth_data['partner_country']))
esquema = construir_esquema(nombre_paleta, colores_paleta, len(paises_grafico))
color_pais = dict(zip(paises_grafico, esquema['series']))   # mismo color por país en los 4 gráficos
print(f"\n🎨 Paleta aplicada: {esquema['nombre']}")

estilo_paleta = {
    'axes.edgecolor': esquema['texto'],
    'axes.labelcolor': esquema['texto'],
    'axes.titlecolor': esquema['texto'],
    'xtick.color': esquema['texto'],
    'ytick.color': esquema['texto'],
    'text.color': esquema['texto'],
    'grid.color': esquema['cuadricula'],
    'axes.facecolor': 'white',
    'figure.facecolor': 'white',
    'legend.edgecolor': esquema['cuadricula'],
}

with plt.rc_context(estilo_paleta):
    # Gráficos mejorados con valores
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f'Market Share Analysis — Paleta: {esquema["nombre"]}', fontsize=16,
                 fontweight='bold', color=esquema['texto'])

    # Market Share Trends con valores en puntos clave
    pivot_share.plot(kind='line', marker='o', ax=axes[0,0], linewidth=2, markersize=6,
                     color=[color_pais[c] for c in pivot_share.columns])
    axes[0,0].set_title('Market Share Evolution (%)', fontsize=14, fontweight='bold')
    axes[0,0].set_ylabel('Market Share (%)')
    axes[0,0].grid(True, alpha=0.6)
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Añadir valores para Brasil y Colombia (principales)
    for country in ['Brasil', 'Colombia']:
        if country in pivot_share.columns:
            for year in pivot_share.index:
                value = pivot_share.loc[year, country]
                if not pd.isna(value):
                    axes[0,0].annotate(f'{value:.1f}%',
                                     (year, value),
                                     textcoords="offset points",
                                     xytext=(0,10),
                                     ha='center', fontsize=8, color=color_pais[country])

    # Trade Value Evolution
    (pivot_value/1000).plot(kind='line', marker='s', ax=axes[0,1], linewidth=2, markersize=6,
                            color=[color_pais[c] for c in pivot_value.columns])
    axes[0,1].set_title('Trade Value Evolution (Million USD)', fontsize=14, fontweight='bold')
    axes[0,1].set_ylabel('Million USD')
    axes[0,1].grid(True, alpha=0.6)

    # Market Share actual con valores
    latest_year = df['year'].max()
    latest_data = market_evolution_top[market_evolution_top['year'] == latest_year].nlargest(8, 'market_share')
    bars = latest_data.set_index('partner_country')['market_share'].plot(
        kind='bar', ax=axes[1,0],
        color=[color_pais[c] for c in latest_data['partner_country']], edgecolor=esquema['texto'])
    axes[1,0].set_title(f'Market Share {latest_year} (%)', fontsize=14, fontweight='bold')
    axes[1,0].set_ylabel('Market Share (%)')
    axes[1,0].tick_params(axis='x', rotation=45)

    # Añadir valores encima de las barras
    for i, (country, value) in enumerate(latest_data.set_index('partner_country')['market_share'].items()):
        axes[1,0].text(i, value + 0.5, f'{value:.1f}%', ha='center', va='bottom', fontsize=9)

    # CAGR con valores
    bars2 = growth_data.set_index('partner_country')['cagr'].plot(
        kind='bar', ax=axes[1,1],
        color=[color_pais[c] for c in growth_data['partner_country']], edgecolor=esquema['texto'])
    axes[1,1].set_title('CAGR % (Compound Annual Growth Rate)', fontsize=14, fontweight='bold')
    axes[1,1].set_ylabel('CAGR (%)')
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].axhline(y=0, color=esquema['texto'], linestyle='-', alpha=0.5)

    # Añadir valores encima de las barras
    for i, (country, value) in enumerate(growth_data.set_index('partner_country')['cagr'].items()):
        axes[1,1].text(i, value + 0.5 if value > 0 else value - 1, f'{value:.1f}%',
                       ha='center', va='bottom' if value > 0 else 'top', fontsize=9)

    # Mostrar los años como números enteros (sin 2021.25, 2021.50...)
    for ax in [axes[0,0], axes[0,1]]:
        ax.set_xticks(pivot_share.index)
        ax.set_xticklabels(pivot_share.index.astype(int))

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
```
4.6. Generar Insights estratégicos o "Descubrimientos relevantes"
Nota: Reemplazar "producto analizado" (Line 2) y "Producto" (Line 35) por el número de partida y nombre del producto analizado antes de proceder con la ejecución del script.
```
# Generar insights automáticos
def generate_insights(df, market_evolution_top, product_name="producto analizado"):
    latest_year = df['year'].max()
    first_year = df['year'].min()

    print(f"📊 INSIGHTS ESTRATÉGICOS - {product_name.upper()}")
    print("="*60)

    # Top 3 países
    top3 = market_evolution_top[market_evolution_top['year'] == latest_year].nlargest(3, 'market_share')
    print(f"\n🏆 TOP 3 PAÍSES ({latest_year}):")
    for i, (_, row) in enumerate(top3.iterrows(), 1):
        print(f"  {i}. {row['partner_country']}: {row['market_share']:.1f}% market share")

    # Mayor crecimiento
    growth_analysis = market_evolution_top.groupby('partner_country').apply(
        lambda x: x['trade_value_usd'].iloc[-1] / x['trade_value_usd'].iloc[0] - 1
    ).sort_values(ascending=False)

    print(f"\n📈 MAYOR CRECIMIENTO ({first_year}-{latest_year}):")
    print(f"  {growth_analysis.index[0]}: +{growth_analysis.iloc[0]:.1%}")

    # Oportunidades
    declining = growth_analysis[growth_analysis < 0]
    if len(declining) > 0:
        print(f"\n⚠️  MERCADOS EN DECLIVE:")
        for country, decline in declining.head(3).items():
            print(f"  {country}: {decline:.1%}")

    # Tamaño total del mercado
    total_market = df[df['year'] == latest_year]['trade_value_usd'].sum()
    print(f"\n💰 TAMAÑO TOTAL DEL MERCADO ({latest_year}): ${total_market/1e9:.1f}B USD")

# Define the product_name variable
product_name = "Producto"

# Aplicar análisis con nombre de producto dinámico
generate_insights(df, market_evolution_top, product_name)
```
4.7. Generación de Promts para análisis posteriores
Nota: Reemplazar "market_name" (Line 7) por el mercado objetivo (país-destino) antes de proceder con la ejecución del script.
```
# GENERAR PROMPT AUTOMÁTICO PARA ANÁLISIS
def generate_analysis_prompt(df, market_evolution_top, product_name):
    latest_year = df['year'].max()
    first_year = df['year'].min()

    # Detectar mercado objetivo (desde columnas o contexto)
    market_name = "Estados Unidos"  # Por defecto, se podría dinamizar

    # Generar estadísticas clave
    top3 = market_evolution_top[market_evolution_top['year'] == latest_year].nlargest(3, 'market_share')
    total_market = df[df['year'] == latest_year]['trade_value_usd'].sum()

    prompt = f"""
    Actúa como consultor en inteligencia de mercados y analiza estos datos de market share de {product_name.lower()} en {market_name} ({first_year}-{latest_year}). Proporciona insights estratégicos específicos:

    CONTEXTO DEL MERCADO:
    - Producto: {product_name}
    - Mercado objetivo: {market_name}
    - Período: {first_year}-{latest_year}
    - Tamaño total del mercado ({latest_year}): ${total_market/1000:.0f} millones USD

    TOP 3 LÍDERES ACTUALES ({latest_year}):
    """

    for i, (_, row) in enumerate(top3.iterrows(), 1):
        prompt += f"\n    {i}. {row['partner_country']}: {row['market_share']:.1f}% market share"

    prompt += f"""

    DATOS PARA ANÁLISIS:
    [Pegar aquí las 4 tablas generadas por el script anterior]

    PREGUNTAS ESPECÍFICAS A RESPONDER:

    1. ANÁLISIS COMPETITIVO:
       - ¿Cuáles son las tendencias de los líderes del mercado?
       - ¿Qué países están ganando/perdiendo participación?
       - ¿Hay oportunidades por concentración/fragmentación del mercado?

    2. OPORTUNIDADES DE ENTRADA:
       - ¿Qué nichos o segmentos presentan menos competencia?
       - ¿Existen países que han reducido su participación creando vacíos?
       - ¿Cuál sería el mejor timing para entrar al mercado?

    3. BENCHMARKING REGIONAL:
       - ¿Cómo se comportan países con perfil similar al de tu empresa?
       - ¿Qué países de América Latina tienen mejor performance?
       - ¿Cuáles son las mejores prácticas observables?

    4. ESTRATEGIA DE POSICIONAMIENTO:
       - ¿Competir directamente con líderes o buscar nichos?
       - ¿Qué ventajas competitivas serían más relevantes?
       - ¿Cuál sería un market share objetivo realista en 2-3 años?

    5. RIESGOS Y AMENAZAS:
       - ¿Qué competidores emergentes representan amenazas?
       - ¿Hay signos de saturación o madurez del mercado?
       - ¿Qué factores externos podrían afectar la industria?

    FORMATO DE RESPUESTA:
    - Insights concisos y accionables
    - Recomendaciones estratégicas específicas
    - Identificación de 2-3 oportunidades prioritarias
    - Métricas clave a monitorear
    """

    return prompt

# Generar y mostrar el prompt
analysis_prompt = generate_analysis_prompt(df, market_evolution_top, product_name)
print("="*80)
print("📋 PROMPT PARA ANÁLISIS ESTRATÉGICO:")
print("="*80)
print(analysis_prompt)
print("\n" + "="*80)
print("💡 INSTRUCCIONES:")
print("1. Copia este prompt")
print("2. Pega las 4 tablas de datos generadas arriba")
print("3. Úsalo con cualquier chatbot para obtener insights estratégicos")
print("="*80)
```
4.8. Helpers (opcionales) para depuración y personalización
4.8.1. Promt para análisis de oportunidad: Origen (Proveedor)-Destino
```
Con base en lo anterior, redacta un texto argumentativo, en tono formal y académico en tercera persona, máximo una cuartilla (250 palabras) donde evidencies los resultados de este estudio de mercados. Enfatiza en el nivel de oportunidad que tendría [COLOMBIA] para ingresar a este mercado desde el punto de vista comercial basándose únicamente en los datos suministrados.
```
4.8.2. Promt para personalizar colores en los gráficos
```
Con base en la siguiente paleta de colores:

[CSS.CODE]

Ajusta el siguiente trozo de código en consecuencia, da por hecho que las librerías ya se encuentran instaladas:

[PY.CODE.4.5]
```
4.8.3. Ajuste para datos faltantes o sin datos históricos (No calcula CAGR)
```
# TABLA 1: Market Share por año y país
print("📊 TABLA 1: MARKET SHARE POR AÑO (%)")
print("="*60)
pivot_share = market_evolution_top.pivot(index='year', columns='partner_country', values='market_share')
pivot_share_rounded = pivot_share.round(1)
print(pivot_share_rounded.to_string())

# TABLA 2: Valores de comercio (Millones USD)
print("\n\n💰 TABLA 2: VALORES DE COMERCIO (Millones USD)")
print("="*60)
pivot_value = market_evolution_top.pivot(index='year', columns='partner_country', values='trade_value_usd')
pivot_value_millions = (pivot_value / 1000).round(1)  # Convertir a millones
print(pivot_value_millions.to_string())

# TABLA 3: Crecimiento anual por país
# El cálculo del CAGR requiere al menos dos años de datos. Como solo se tiene un año (2024),
# esta tabla no se puede generar con los datos actuales.
# print("\n\n📈 TABLA 3: CRECIMIENTO ANUAL COMPUESTO (CAGR %)")
# print("="*60)
# growth_data = market_evolution_top.groupby('partner_country').apply(
#     lambda x: ((x['trade_value_usd']. iloc[-1] / x['trade_value_usd'].iloc[0]) ** (1/(len(x)-1)) - 1) * 100
# ).reset_index(name='cagr')
# growth_summary = growth_data.set_index('partner_country')['cagr'].round(1).sort_values(ascending=False)
# print(growth_summary.to_string())
print("\n\n📈 TABLA 3: CRECIMIENTO ANUAL COMPUESTO (CAGR %)")
print("="*60)
print("No se puede calcular el CAGR. Se requiere al menos dos años de datos para este cálculo.")

# TABLA 4: Ranking por año
print("\n\n🏆 TABLA 4: RANKING DE PAÍSES POR AÑO (Top 5)")
print("="*60)
for year in sorted(df['year'].unique()):
    year_data = market_evolution_top[market_evolution_top['year'] == year].nlargest(5, 'market_share')
    print(f"\n{year}:")
    for i, (_, row) in enumerate(year_data.iterrows(), 1):
        print(f"  {i}. {row['partner_country']}: {row['market_share']:.1f}% ({row['trade_value_usd']/1000:.0f}M USD)")
```
4.8.4. Ajuste para datos faltantes (Calcula CAGR omitiendo datos vacíos o nulos)
```
# TABLA 1: Market Share por año y país
print("📊 TABLA 1: MARKET SHARE POR AÑO (%)")
print("="*60)
pivot_share = market_evolution_top.pivot(index='year', columns='partner_country', values='market_share')
pivot_share_rounded = pivot_share.round(1)
print(pivot_share_rounded.to_string())

# TABLA 2: Valores de comercio (Millones USD)
print("\n\n💰 TABLA 2: VALORES DE COMERCIO (Millones USD)")
print("="*60)
pivot_value = market_evolution_top.pivot(index='year', columns='partner_country', values='trade_value_usd')
pivot_value_millions = (pivot_value / 1000).round(1)  # Convertir a millones
print(pivot_value_millions.to_string())

# TABLA 3: Crecimiento anual por país
print("\n\n📈 TABLA 3: CRECIMIENTO ANUAL COMPUESTO (CAGR %)")
print("="*60)
growth_data = market_evolution_top.groupby('partner_country').apply(
    lambda x: ((x['trade_value_usd'].iloc[-1] / x['trade_value_usd'].iloc[0]) ** (1/(len(x)-1)) - 1) * 100 if len(x) > 1 else np.nan
).reset_index(name='cagr')
growth_summary = growth_data.set_index('partner_country')['cagr'].round(1).sort_values(ascending=False)
print(growth_summary.to_string())

# TABLA 4: Ranking por año
print("\n\n🏆 TABLA 4: RANKING DE PAÍSES POR AÑO (Top 5)")
print("="*60)
for year in sorted(df['year'].unique()):
    year_data = market_evolution_top[market_evolution_top['year'] == year].nlargest(5, 'market_share')
    print(f"\n{year}:")
    for i, (_, row) in enumerate(year_data.iterrows(), 1):
        print(f"  {i}. {row['partner_country']}: {row['market_share']:.1f}% ({row['trade_value_usd']/1000:.0f}M USD)")
```
