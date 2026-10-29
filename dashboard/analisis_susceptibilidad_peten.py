"""
Predicción de zonas susceptibles a incendios forestales en Petén
Análisis geoespacial de los rásters de susceptibilidad y validación con focos FIRMS.

Qué hace este script
--------------------
1. Estadísticas de cada ráster de susceptibilidad (2015–2025 y modelos finales):
   media, mediana, percentil 90, área por clase y histograma.
2. Cambio 2015 → 2025 y persistencia (años con valor >= 0.6).
3. Resumen por celda de la cuadrícula de 5 km y por municipio.
4. Validación con focos de calor FIRMS (VIIRS S-NPP): % de focos en cada clase,
   % en alta + muy alta, AUC, y prueba del modelo RF 2017–2024 con años fuera
   de su periodo.
5. Exporta tablas CSV (para la tesis) y los datos que usa el dashboard web.

Requisitos
----------
    pip install numpy pillow
No necesita GDAL ni rasterio: los GeoTIFF se leen con Pillow y los shapefiles
con un lector propio (más abajo). Todos los archivos deben estar en
WGS 84 / UTM zona 16N (EPSG:32616), igual que los exportaste de QGIS.

Uso
---
    python analisis_susceptibilidad_peten.py --datos "C:/ruta/a/mis_datos" --salida "C:/ruta/salida"

En la carpeta --datos deben estar (con estos nombres):
    susceptibilidad_2015.tif ... susceptibilidad_2025.tif
    susceptibilidad_final_2015_2025.tif
    susceptibilidad_final_2017_2024.tif
    susceptibilidad_final_2017_2024_RF.tif
    cuadricula_5km_peten_recortada_2017_2024.shp/.dbf
    zonas_5km_municipios_2017_2024.dbf              (municipio de cada celda)
    susceptibilidad_municipios_2017_2024.shp/.dbf
    zonas_maximas_por_municipio_2017_2024.dbf
    firms_peten_2015_2025_utm.shp/.dbf              (focos FIRMS 2015–2025)
Si falta alguna capa vectorial o FIRMS, esa parte se omite y el resto continúa.

Clases (intervalos iguales sobre 0–1, como en QGIS):
    Muy baja < 0.2 | Baja 0.2–0.4 | Media 0.4–0.6 | Alta 0.6–0.8 | Muy alta >= 0.8
"""
import argparse, csv, json, math, os, struct, sys
import numpy as np
from PIL import Image, ImageDraw

Image.MAX_IMAGE_PIXELS = None

# ---------------------------------------------------------------- configuración
ANIOS = [str(a) for a in range(2015, 2026)]
FINALES = ['final_2015_2025', 'final_2017_2024', 'final_2017_2024_RF']
CAPAS = ANIOS + FINALES
NOMBRE = {**{a: f'Año {a}' for a in ANIOS},
          'final_2015_2025': 'Final 2015–2025', 'final_2017_2024': 'Final 2017–2024',
          'final_2017_2024_RF': 'Final 2017–2024 (RF)'}
CLASES = ['Muy baja', 'Baja', 'Media', 'Alta', 'Muy alta']
CORTES = [0, 0.2, 0.4, 0.6, 0.8, 1.0000001]
UMBRAL_ALTA = 0.6
NODATA = -9999
MODELO_RF = 'final_2017_2024_RF'
PERIODO_RF = range(2017, 2025)          # años con los que se entrenó el RF


# ---------------------------------------------------------------- lectura de datos
def leer_tif(ruta):
    """Lee un GeoTIFF de una banda (float32) y su georreferencia (esquina sup. izq. y tamaño de píxel)."""
    im = Image.open(ruta)
    a = np.asarray(im, dtype=np.float32)
    tags = im.tag_v2
    escala = tags.get(33550)            # ModelPixelScaleTag
    punto = tags.get(33922)             # ModelTiepointTag
    if not escala or not punto:
        sys.exit(f'{ruta} no tiene georreferencia GeoTIFF')
    return a, dict(x0=punto[3], y0=punto[4], res=escala[0], alto=a.shape[0], ancho=a.shape[1])


def leer_shp(ruta):
    """Lee un shapefile de puntos (tipo 1) o polígonos (tipo 5). Devuelve lista de geometrías."""
    b = open(ruta, 'rb').read()
    pos, geoms = 100, []
    while pos < len(b):
        _, largo = struct.unpack('>2i', b[pos:pos + 8]); pos += 8
        rec = b[pos:pos + largo * 2]; pos += largo * 2
        tipo = struct.unpack('<i', rec[:4])[0]
        if tipo == 0:
            geoms.append(None)
        elif tipo == 1:
            geoms.append(struct.unpack('<2d', rec[4:20]))
        else:
            npartes, npuntos = struct.unpack('<2i', rec[36:44])
            partes = list(struct.unpack(f'<{npartes}i', rec[44:44 + 4 * npartes]))
            o = 44 + 4 * npartes
            xy = struct.unpack(f'<{2 * npuntos}d', rec[o:o + 16 * npuntos])
            pts = [(xy[2 * i], xy[2 * i + 1]) for i in range(npuntos)]
            geoms.append([pts[partes[i]:(partes[i + 1] if i + 1 < npartes else npuntos)] for i in range(npartes)])
    return geoms


def leer_dbf(ruta):
    """Lee la tabla de atributos (.dbf). Usa el .cpg vecino para la codificación."""
    cpg = os.path.splitext(ruta)[0] + '.cpg'
    cod = open(cpg).read().strip() if os.path.exists(cpg) else 'cp1252'
    cod = {'1252': 'cp1252', 'UTF-8': 'utf-8'}.get(cod, cod)
    b = open(ruta, 'rb').read()
    n, largo_enc, largo_reg = struct.unpack('<IHH', b[4:12])
    campos, p = [], 32
    while b[p] != 0x0D:
        nombre = b[p:p + 11].split(b'\0')[0].decode(); tipo = chr(b[p + 11]); largo = b[p + 16]
        campos.append((nombre, tipo, largo)); p += 32
    filas = []
    for i in range(n):
        r = b[largo_enc + i * largo_reg: largo_enc + (i + 1) * largo_reg]
        if r[:1] == b'*':
            continue
        o, fila = 1, {}
        for nombre, tipo, largo in campos:
            v = r[o:o + largo].decode(cod, 'replace').strip(); o += largo
            if tipo in 'NF':
                try: v = float(v) if v else None
                except ValueError: v = None
            fila[nombre] = v
        filas.append(fila)
    return filas


def area_firmada(anillo):
    return 0.5 * sum(anillo[i][0] * anillo[i - 1][1] - anillo[i - 1][0] * anillo[i][1] for i in range(len(anillo)))


def rasterizar_poligonos(geoms, g):
    """Devuelve una matriz del tamaño del ráster con el índice del polígono de cada píxel (-1 = ninguno)."""
    lienzo = Image.new('I', (g['ancho'], g['alto']), 0)
    d = ImageDraw.Draw(lienzo)
    a_px = lambda r: [((x - g['x0']) / g['res'] - 0.5, (g['y0'] - y) / g['res'] - 0.5) for x, y in r]
    orden = sorted(range(len(geoms)), key=lambda i: -sum(abs(area_firmada(r)) for r in geoms[i]))
    for i in orden:                       # los más grandes primero, así los enclaves quedan encima
        for r in geoms[i]:
            if area_firmada(r) > 0: d.polygon(a_px(r), fill=i + 1)   # anillo exterior
        for r in geoms[i]:
            if area_firmada(r) < 0: d.polygon(a_px(r), fill=0)       # hueco
    return np.asarray(lienzo, dtype=np.int32) - 1


def clase_de(v):
    return np.clip(np.digitize(v, CORTES) - 1, 0, 4)


def auc(valores_foco, valores_area, rng):
    """AUC = probabilidad de que un foco tenga mayor susceptibilidad que un píxel al azar."""
    muestra = np.sort(rng.choice(valores_area, min(3_000_000, valores_area.size), replace=False))
    lo = np.searchsorted(muestra, valores_foco, 'left'); hi = np.searchsorted(muestra, valores_foco, 'right')
    return float(((lo + hi) / 2).mean() / muestra.size)


def escribir_csv(ruta, encabezado, filas):
    with open(ruta, 'w', newline='', encoding='utf-8-sig') as f:   # utf-8-sig: Excel abre bien las tildes
        w = csv.writer(f); w.writerow(encabezado); w.writerows(filas)
    print('  ->', os.path.basename(ruta))


# ---------------------------------------------------------------- análisis
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--datos', required=True, help='carpeta con los .tif y .shp')
    ap.add_argument('--salida', default='salida_peten', help='carpeta de resultados')
    args = ap.parse_args()
    D = lambda n: os.path.join(args.datos, n)
    os.makedirs(args.salida, exist_ok=True)
    S = lambda n: os.path.join(args.salida, n)
    rng = np.random.default_rng(42)

    # 1) Estadísticas por capa ------------------------------------------------
    print('1/4 Estadísticas por capa')
    stats, g, valido_todos, cuenta_alta = {}, None, None, None
    primera = ultima = None
    for k in CAPAS:
        a, gk = leer_tif(D(f'susceptibilidad_{k}.tif'))
        g = g or gk
        m = a != NODATA; v = a[m]
        cls = np.bincount(clase_de(v), minlength=5)
        km2 = g['res'] ** 2 / 1e6
        stats[k] = dict(media=float(v.mean()), desv=float(v.std()), mediana=float(np.median(v)),
                        p90=float(np.percentile(v, 90)), n=int(v.size), area_km2=v.size * km2,
                        cls_km2=(cls * km2).tolist(), cls_pct=(cls / v.size * 100).tolist(),
                        hist=(np.histogram(v, bins=50, range=(0, 1))[0] / v.size).tolist())
        if k in ANIOS:
            valido_todos = m if valido_todos is None else valido_todos & m
            alta = (a >= UMBRAL_ALTA) & m
            cuenta_alta = alta.astype(np.uint8) if cuenta_alta is None else cuenta_alta + alta
        if k == ANIOS[0]: primera = a.copy()
        if k == ANIOS[-1]: ultima = a
        print(f'   {NOMBRE[k]:24s} media {stats[k]["media"]:.3f}   alta+muy alta {stats[k]["cls_pct"][3] + stats[k]["cls_pct"][4]:.1f}%')
    escribir_csv(S('estadisticas_por_capa.csv'),
                 ['capa', 'media', 'desviacion', 'mediana', 'p90', 'area_km2'] + [f'{c} km2' for c in CLASES] + [f'{c} %' for c in CLASES] + ['alta+muy alta %'],
                 [[NOMBRE[k], round(s['media'], 4), round(s['desv'], 4), round(s['mediana'], 4), round(s['p90'], 4), round(s['area_km2'], 1)]
                  + [round(x, 1) for x in s['cls_km2']] + [round(x, 2) for x in s['cls_pct']] + [round(s['cls_pct'][3] + s['cls_pct'][4], 2)]
                  for k, s in stats.items()])

    # cambio y persistencia
    dif = (ultima - primera)[valido_todos]
    pers = cuenta_alta[valido_todos]
    km2 = g['res'] ** 2 / 1e6
    escribir_csv(S('cambio_2015_2025.csv'), ['categoria', '% del area'],
                 [['Aumentó (> +0.1)', round(float((dif > 0.1).mean() * 100), 2)],
                  ['Estable (±0.1)', round(float((abs(dif) <= 0.1).mean() * 100), 2)],
                  ['Disminuyó (< −0.1)', round(float((dif < -0.1).mean() * 100), 2)],
                  ['Cambio medio', round(float(dif.mean()), 4)]])
    cp = np.bincount(pers, minlength=12)[:12]
    escribir_csv(S('persistencia_alta.csv'), ['años con valor >= 0.6', 'km2', '% del area'],
                 [[i, round(c * km2, 1), round(c / pers.size * 100, 2)] for i, c in enumerate(cp)])
    del primera, ultima, dif

    # 2) Municipios -------------------------------------------------------------
    print('2/4 Municipios')
    L = None
    if os.path.exists(D('susceptibilidad_municipios_2017_2024.shp')):
        mg = leer_shp(D('susceptibilidad_municipios_2017_2024.shp'))
        nombres = [r['shapeName'] for r in leer_dbf(D('susceptibilidad_municipios_2017_2024.dbf'))]
        L = rasterizar_poligonos(mg, g).ravel()
        NM = len(nombres)
        mun = {k: {} for k in CAPAS}
        for k in CAPAS:
            a = leer_tif(D(f'susceptibilidad_{k}.tif'))[0].ravel()
            m = (a != NODATA) & (L >= 0); li, v = L[m], a[m]
            n = np.bincount(li, minlength=NM)
            mun[k]['media'] = np.bincount(li, weights=v, minlength=NM) / n
            mun[k]['alta'] = np.bincount(li, weights=(v >= UMBRAL_ALTA), minlength=NM) / n * 100
            mun[k]['cls'] = np.bincount(li * 5 + clase_de(v), minlength=NM * 5).reshape(NM, 5)
        filas = []
        for j, nm in enumerate(nombres):
            c = mun[MODELO_RF]['cls'][j]
            filas.append([nm, round(c.sum() * km2, 1), round(float(mun[MODELO_RF]['media'][j]), 4), round(float(mun[MODELO_RF]['alta'][j]), 2)]
                         + [round(x * km2, 1) for x in c]
                         + [round(float(mun[a]['alta'][j]), 2) for a in ANIOS])
        filas.sort(key=lambda r: -r[3])
        escribir_csv(S('municipios.csv'),
                     ['municipio', 'area_km2', 'media RF', '% alta+muy alta RF'] + [f'{c} km2 RF' for c in CLASES] + [f'% alta {a}' for a in ANIOS], filas)
    else:
        print('   (no se encontró la capa de municipios; se omite)')

    # 3) Cuadrícula de 5 km -----------------------------------------------------
    print('3/4 Cuadrícula de 5 km')
    if os.path.exists(D('cuadricula_5km_peten_recortada_2017_2024.dbf')):
        celdas = leer_dbf(D('cuadricula_5km_peten_recortada_2017_2024.dbf'))
        mun_celda = {}
        if os.path.exists(D('zonas_5km_municipios_2017_2024.dbf')):
            mun_celda = {int(r['id']): r['mun_shapeN'] for r in leer_dbf(D('zonas_5km_municipios_2017_2024.dbf'))}
        gx0 = min(r['left'] for r in celdas); gy0 = max(r['top'] for r in celdas)
        tam = celdas[0]['right'] - celdas[0]['left']
        ncol = max(int(r['col_index']) for r in celdas) + 1; nfil = max(int(r['row_index']) for r in celdas) + 1
        # índice de celda de cada píxel según su centro
        xs = g['x0'] + (np.arange(g['ancho']) + 0.5) * g['res']; ys = g['y0'] - (np.arange(g['alto']) + 0.5) * g['res']
        cc = np.floor((xs - gx0) / tam).astype(int); ff = np.floor((gy0 - ys) / tam).astype(int)
        tabla = np.full(nfil * ncol, -1, np.int32)
        for i, r in enumerate(celdas): tabla[int(r['row_index']) * ncol + int(r['col_index'])] = i
        dentro = (cc[None, :] >= 0) & (cc[None, :] < ncol) & (ff[:, None] >= 0) & (ff[:, None] < nfil)
        C = np.where(dentro, tabla[np.clip(ff[:, None] * ncol + cc[None, :], 0, nfil * ncol - 1)], -1).ravel()
        NC = len(celdas); res_c = {}
        for k in CAPAS:
            a = leer_tif(D(f'susceptibilidad_{k}.tif'))[0].ravel()
            m = (a != NODATA) & (C >= 0); ci, v = C[m], a[m]
            n = np.bincount(ci, minlength=NC)
            with np.errstate(invalid='ignore', divide='ignore'):
                res_c[k] = (np.bincount(ci, weights=v, minlength=NC) / n, np.bincount(ci, weights=(v >= UMBRAL_ALTA), minlength=NC) / n * 100, n)
        filas = []
        for i, r in enumerate(celdas):
            media, alta, n = res_c[MODELO_RF][0][i], res_c[MODELO_RF][1][i], res_c[MODELO_RF][2][i]
            if n == 0: continue
            filas.append([int(r['id']), mun_celda.get(int(r['id']), ''), int(r['row_index']), int(r['col_index']), round(n * km2, 2),
                          round(float(media), 4), round(float(alta), 2)] + [round(float(res_c[a][0][i]), 4) for a in ANIOS])
        filas.sort(key=lambda r: -r[6])
        escribir_csv(S('celdas_5km.csv'), ['id', 'municipio', 'fila', 'columna', 'area_km2', 'media RF', '% alta+muy alta RF'] + [f'media {a}' for a in ANIOS], filas)
        del C
    else:
        print('   (no se encontró la cuadrícula; se omite)')

    # 4) Validación con FIRMS -------------------------------------------------------
    print('4/4 Validación con focos FIRMS')
    if os.path.exists(D('firms_peten_2015_2025_utm.shp')):
        P = np.array([p if p else (np.nan, np.nan) for p in leer_shp(D('firms_peten_2015_2025_utm.shp'))])
        filas_f = leer_dbf(D('firms_peten_2015_2025_utm.dbf'))
        anio = np.array([r['acq_date'][:4] for r in filas_f]); tipo = np.array([r['type'] for r in filas_f])
        col = np.floor((P[:, 0] - g['x0']) / g['res']); fil = np.floor((g['y0'] - P[:, 1]) / g['res'])
        ok = (col >= 0) & (col < g['ancho']) & (fil >= 0) & (fil < g['alto']) & (tipo == 0)   # type 0 = incendio de vegetación
        col = np.clip(np.nan_to_num(col), 0, g['ancho'] - 1).astype(int); fil = np.clip(np.nan_to_num(fil), 0, g['alto'] - 1).astype(int)
        rf = leer_tif(D(f'susceptibilidad_{MODELO_RF}.tif'))[0]
        rf_area = rf[rf != NODATA]
        filas_v = []
        for a in ANIOS:
            sel = ok & (anio == a)
            if not sel.any(): continue
            mapa = leer_tif(D(f'susceptibilidad_{a}.tif'))[0]
            v = mapa[fil[sel], col[sel]]; vr = rf[fil[sel], col[sel]]
            dentro_mapa = v != NODATA; v, vr = v[dentro_mapa], vr[dentro_mapa]
            c = np.bincount(clase_de(v), minlength=5); cr = np.bincount(clase_de(vr), minlength=5)
            area_alta = stats[a]['cls_pct'][3] + stats[a]['cls_pct'][4]
            fuera = 'sí' if int(a) not in PERIODO_RF else 'no'
            filas_v.append([a, int(v.size), round(float((v >= UMBRAL_ALTA).mean() * 100), 2), round(area_alta, 2), round(auc(v, mapa[mapa != NODATA], rng), 4),
                            round(float((vr >= UMBRAL_ALTA).mean() * 100), 2), round(auc(vr, rf_area, rng), 4), fuera]
                           + [round(x / v.size * 100, 2) for x in c] + [round(x / v.size * 100, 2) for x in cr])
            print(f'   {a}: {v.size:6d} focos | mapa del año {filas_v[-1][2]:5.1f}% en alta (AUC {filas_v[-1][4]:.3f}) | RF {filas_v[-1][5]:5.1f}% (AUC {filas_v[-1][6]:.3f}){"  <- fuera del periodo RF" if fuera == "sí" else ""}')
        escribir_csv(S('validacion_firms.csv'),
                     ['año', 'focos', '% focos en alta+muy alta (mapa del año)', '% área en alta+muy alta (mapa del año)', 'AUC mapa del año',
                      '% focos en alta+muy alta (RF 2017-2024)', 'AUC RF', 'fuera del periodo RF']
                     + [f'% focos {c} (mapa del año)' for c in CLASES] + [f'% focos {c} (RF)' for c in CLASES], filas_v)
        if L is not None:
            Lm = L.reshape(g['alto'], g['ancho'])
            filas_m = []
            for j, nm in enumerate(nombres):
                fila = [nm]
                for a in ANIOS:
                    sel = ok & (anio == a)
                    idx = Lm[fil[sel], col[sel]]
                    fila.append(int((idx == j).sum()))
                filas_m.append(fila + [sum(fila[1:])])
            escribir_csv(S('focos_por_municipio.csv'), ['municipio'] + ANIOS + ['total'], filas_m)
    else:
        print('   (no se encontró firms_peten_2015_2025_utm.shp; se omite)')

    json.dump({k: {kk: (vv if not isinstance(vv, float) else round(vv, 5)) for kk, vv in s.items()} for k, s in stats.items()},
              open(S('estadisticas.json'), 'w'), ensure_ascii=False)
    print(f'\nListo. Resultados en: {os.path.abspath(args.salida)}')


if __name__ == '__main__':
    main()
