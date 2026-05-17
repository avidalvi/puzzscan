mira docs/prd.md

PRD de esta fase creado en: `docs/PRD_obtencion_perfil_pieza.md`

vamos a sacar el perfil del contorno:
- caras 1-2-3-4-5-1
- cada cara va de esqiuina a esquina de la pieza
- la curva de contorno con las entradas y salidas que sigan fielmente la pieza
- la curva tiene un eje central cero sobre la linea que unira las esquinas
- machos son positivos en Y, hembras negativas

número fijo de puntos por cara?

despues una reduccion dimensional de la curva a sus componentes que la representen de forma exacta, pero simplificada

codigo y notebook de exploracion desde la pieza, la curva, la reduccion dimensional, calidad, etc

se guarda en la bbdds: coordenadas iniciales, reduccion dimensional, perfil de cada cara que puede ser: lisa/macho/hembra. todo en el formato 1-...-1

ese perfil reducido se usarà para comparar con los huecos del puzzle y buscar la pieza.






