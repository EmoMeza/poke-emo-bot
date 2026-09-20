# Cómo funciona poke-emo-bot

Un bot que aprende a jugar batallas dobles de Pokémon Showdown (formato
`gen9championsvgc2026regmc`) por **aprendizaje por refuerzo**: juega miles de
partidas contra sí mismo en un servidor local y ajusta una red neuronal (MLP)
con **PPO** para que las jugadas que llevan a ganar sean más probables.

Este documento explica cada pieza: qué hace, por qué está hecha así y un
ejemplo chico. Está pensado para releerlo cuando se olvide la lógica.

---

## Índice

1. Vista general
2. Puesta en marcha
3. Equipos (`teams/`)
4. Observación: qué "ve" la red (`observations/embedding.py`)
5. Acciones y máscaras (`envs/actions.py`)
6. Reward: cómo se puntúa (`rewards/reward.py`)
7. El entorno y el self-play (`envs/vgc_env.py`)
8. La red actor-crítico (`agents/actor_critic.py`)
9. PPO paso a paso (`agents/ppo.py`)
10. El loop de entrenamiento (`train.py`)
11. Jugar contra la IA (`play.py`)
12. Limitaciones y próximos pasos
13. Problemas frecuentes
14. Glosario

---

## 1. Vista general

```
   equipos (teams/)            observación (observations/)      acciones legales (envs/actions.py)
 ┌────────────────────┐        ┌───────────────────────┐        ┌───────────────────────┐
 │ oficial.txt        │        │ embed_battle()        │        │ legal_action_masks()  │
 │ meta/1..5.txt      │        │ batalla → 430 números │        │ second_slot_mask()    │
 └─────────┬──────────┘        └───────────┬───────────┘        └───────────┬───────────┘
           ▼                               ▼                                ▼
 ┌───────────────────────────────────────────────────────────────────────────────────────┐
 │ VGCDoublesEnv (envs/vgc_env.py): dos agentes peleando en el servidor Showdown local   │
 │   agent1 = equipo oficial            agent2 = un equipo meta al azar por partida      │
 └───────────────────────────────────────────┬───────────────────────────────────────────┘
                                             │ observación, reward y máscaras de cada agente
                                             ▼
                                 ActorCritic (MLP, agents/)  ──►  acciones
                                             ▲
                        train.py: PPO ajusta los pesos  ──►  checkpoints/*.pt  ──►  play.py
```

**Estructura del repositorio**

| Ruta | Qué contiene |
|---|---|
| `teams/` | Equipos (`data/oficial.txt`, `data/meta/*.txt`), su carga y un parche a la data de poke-env |
| `observations/embedding.py` | `embed_battle`: convierte el estado de la batalla en un vector de 430 números |
| `rewards/reward.py` | `calc_reward`: el puntaje de cada turno |
| `envs/actions.py` | Las 107 acciones por pokémon y qué acciones son legales en cada turno |
| `envs/vgc_env.py` | `VGCDoublesEnv`: el entorno de entrenamiento (self-play) |
| `agents/actor_critic.py` | La red neuronal y el muestreo de acciones |
| `agents/ppo.py` | El algoritmo: buffer, ventajas (GAE) y actualización PPO |
| `agents/checkpoint.py` | Guardar/cargar pesos |
| `train.py` | Loop de entrenamiento |
| `play.py` | Jugar contra la IA desde el navegador |
| `test_env.py` | Prueba rápida del entorno con jugadas legales al azar |
| `checkpoints/` | Pesos: `pretrained.pt` (publicado en el repo); `best.pt` y `latest.pt` son los de tus corridas locales y no se suben |
| `scripts/` | `setup_server.sh` (instala el servidor local) y `plot_training.py` (gráfico de resultados) |
| `runs/` | Curvas de TensorBoard (no se sube a git) |
| `pokemonshowdown-server/` | El servidor local de Showdown (no se sube a git) |

---

## 2. Puesta en marcha

**Servidor.** `bash scripts/setup_server.sh` lo descarga en `pokemonshowdown-server/`
(ignorado por git) y deja activadas las dos opciones de `config/config.js` que se
necesitan (si lo instalas a mano, actívalas tú). **Hay que reiniciar el servidor
tras cambiarlas**:

```js
exports.noguestsecurity = true;   // permite nombres de usuario cualquiera, sin login real
exports.nothrottle = true;        // sin esto, un 2º desafío antes de 10 s se cancela
                                  // y las partidas seguidas se cuelgan
```

```bash
cd pokemonshowdown-server && ./pokemon-showdown start     # queda escuchando en localhost:8000
```

**Python** (venv del proyecto):

```bash
pip install -r requirements.txt
```

**Comandos** (desde la raíz del repo, con el servidor corriendo):

```bash
python test_env.py            # 3 partidas con jugadas legales al azar: prueba que todo conecta
python train.py               # entrenar (Ctrl+C corta; los pesos se guardan cada 5 updates)
tensorboard --logdir runs     # ver las curvas en http://localhost:6006
python play.py                # jugar contra la IA (ver sección 11)
```

---

## 3. Equipos (`teams/`)

### Formato Stat Points

El formato usa **Stat Points (SP)** en lugar de EVs: máximo 32 por stat, 66 en
total, y todos los IVs fijos en 31 (no se escriben líneas `IVs:`). El servidor
lee los números de la línea `EVs:` **como SP**, así que hay que convertir los
equipos que vienen con EVs:

| Origen | Conversión | Ejemplo |
|---|---|---|
| Sets que suman 264 por pokémon (= 66 × 4) | dividir por 4 | `128 SpA` → `32 SpA` |
| Sets en múltiplos de 8 (EVs clásicos) | dividir por 8 | `252 Spe` → `32 Spe` (máx.) |

Para comprobar que un equipo es válido (salida vacía = válido):

```bash
cd pokemonshowdown-server
./pokemon-showdown validate-team gen9championsvgc2026regmc < ../teams/data/oficial.txt
```

Las megas se escriben como **especie base + mega stone** (`Froslass @ Froslassite`,
no `Froslass-Mega`), que es como Showdown las exporta.

### `loader.py`

- `cargar_equipo_oficial()`: texto del equipo oficial (el que entrena nuestro agente).
- `cargar_equipo_meta(indice=None)`: un equipo meta (al azar si no se da índice).
- `MetaPoolTeambuilder`: entrega un equipo meta al azar **cada vez que arranca
  una batalla**. poke-env le pide el equipo a cada jugador al iniciar cada
  partida, así que basta asignárselo al rival una vez y rota solo entre los
  archivos de `data/meta/`. Así el agente no se acostumbra a un único rival.

### Parche a la data de poke-env (`pokedex_patch.py`)

poke-env 0.10.0 (la última versión) no conoce las megas nuevas (Dragonite,
Floette, Froslass, Scovillain…). Cuando una evolucionaba en plena batalla,
poke-env lanzaba `KeyError` al procesar el mensaje del servidor y **la batalla
se colgaba**. `data/extra_pokedex.json` trae las 88 especies que faltan
(generadas desde la data del propio servidor) y `aplicar_parche_pokedex()` las
agrega al pokedex de poke-env. Se ejecuta sola al importar `teams.loader`.

---

## 4. Observación: qué "ve" la red (`observations/embedding.py`)

La red solo entiende números. `embed_battle(battle)` convierte el objeto
`DoubleBattle` de poke-env en un vector de **430 números** (`float32`) desde la
perspectiva del agente.

### Idea de diseño

No se usa un one-hot de la especie (cientos de columnas casi siempre en cero).
En vez de "quién es" cada pokémon se describe **cómo está** (HP, estado, boosts,
stats, tipos) y **qué puede hacer** (poder, precisión y efectividad de cada
movimiento). Así la red generaliza a pokémon que no vio: aprende "un movimiento
de 90 de poder súper efectivo contra el rival es bueno", no "Salamence hace X".

### Composición

| Bloque | Cantidad | Tamaño | Total |
|---|---|---|---|
| Pokémon activos (2 nuestros + 2 rivales) | 4 | 62 | 248 |
| Banca (2 nuestros + 2 rivales) | 4 | 15 | 60 |
| Movimientos de nuestros 2 activos (4 movs × 5 datos) | 2 | 20 | 40 |
| Banderas: `can_dynamax`, `can_mega_evolve`, `can_tera` (×2 slots), mega ya usada (nosotros y rival), `trapped` ×2, `force_switch` ×2 | | | 12 |
| Clima | | | 9 |
| Campos (terrenos, Trick Room, gravedad…) | | | 13 |
| Condiciones de lado (pantallas, Tailwind, trampas…): nuestro lado + rival | 2 | 24 | 48 |
| **Total** | | | **430** |

**Pokémon activo (62 números):**
`HP (1) | estado one-hot (7) | boosts (7) | stats (6) | tipo 1 one-hot (20) | tipo 2 one-hot (20) | ¿es mega? (1)`

**Banca (15 números):** `HP (1) | estado (7) | stats (6) | ¿es mega? (1)`

**Movimiento (5 números):** `poder | precisión | categoría | efectividad vs rival 1 | efectividad vs rival 2`

### Escalas

Todo se lleva a un rango parecido para que la red aprenda estable: HP como
fracción 0–1, boosts ÷ 6 (rango −1 a 1), stats ÷ 200, poder ÷ 200 y
efectividad ÷ 4.

### Ejemplo: la efectividad de un movimiento

`Pokemon.damage_multiplier(move)` resume toda la tabla de tipos en un número.
Ice Fang (Hielo) contra un Garchomp (Dragón/Tierra) hace ×4 (débil a Hielo por
ambos tipos), y en el vector queda `4 / 4 = 1.0`. Contra un rival inmune sería
`0.0`. Como hay dos rivales activos, cada movimiento lleva dos de estos números.

### Información oculta: qué sabe realmente el agente

- **Nuestros pokémon**: se usan las stats reales (con SP, nature y nivel), porque
  las conocemos.
- **Los del rival**: solo `base_stats` de la especie. No conocemos su inversión
  ni su nature; es lo mismo que sabría un jugador humano.
- **Banca del rival**: solo los pokémon que ya aparecieron en batalla. Los que
  no han salido se representan con ceros.
- **Banca nuestra**: `battle.team` trae siempre los 6 pokémon aunque solo se
  lleven 4. El request del servidor (`battle.last_request`) lista únicamente los
  llevados; se filtra con eso para que los 2 que quedaron fuera no aparezcan
  como "banca sana" para siempre.

### Mega evolución

Cambia stats, tipos y habilidad, y solo se puede una vez por partida. poke-env
no expone un "¿es mega?", y lo guarda distinto según el lado: en **nuestros**
pokémon la especie pasa a ser la forma mega, en los del **rival** conserva la
especie base pero cambia sus `base_stats`. `_is_mega` revisa ambos casos. Además
las banderas de batalla `used_mega_evolve` / `opponent_used_mega_evolve` dicen si
la mega ya se gastó.

---

## 5. Acciones y máscaras (`envs/actions.py`)

### Las 107 acciones de cada slot

En dobles hay dos pokémon activos ("slots"), y cada uno elige entre 107 acciones:

| Índice | Significado |
|---|---|
| `0` | Pasar (solo si no hay ninguna otra opción) |
| `1`–`6` | Cambiar al pokémon k de `battle.team` |
| `7`–`26` | Movimiento 1–4, cada uno con 5 objetivos posibles |
| `27`–`106` | Lo mismo, con mega / z-move / dynamax / tera |

Para `a ≥ 7`:

```
movimiento = ((a − 7) % 20) // 5           # 0..3
objetivo   = ((a − 7) %  5) − 2            # −2, −1, 0, 1, 2
gimmick    = (a − 7) // 20                 # 0 nada, 1 mega, 2 z, 3 dynamax, 4 tera
```

Objetivos: `−1` y `−2` son tus dos pokémon (`−1` el slot 1, `−2` el slot 2), `1` y `2`
son los dos rivales, `0` es "sin objetivo".

**Ejemplo:** la acción `10` es `(10 − 7) = 3` → movimiento 1, objetivo `3 % 5 − 2 = 1`:
"usa el primer movimiento contra el primer rival". La acción `27 + 10 = 37` sería
lo mismo con mega evolución.

La acción completa de un turno es un par `[a1, a2]`, una por slot.

### Por qué hacen falta máscaras

Cada turno casi todas las 107 acciones son ilegales (movimiento sin PP, objetivo
inválido, cambio a un pokémon debilitado o no llevado…). Sin máscaras, la red
gastaría buena parte del entrenamiento aprendiendo a no elegirlas.

`legal_action_masks(battle)` devuelve un arreglo booleano `(2, 107)` con las
acciones legales de cada slot. Reutiliza las mismas comprobaciones que poke-env
hace al convertir una acción en orden del servidor: si `_action_to_order_individual`
levanta `AssertionError`, la acción es ilegal.

**Aplicar una máscara**: a los logits de las acciones ilegales se les pone
`−1e9` antes del softmax, y su probabilidad queda en 0.

```
logits    = [2.0,  1.0,  0.5]
legales   = [sí,   no,   sí ]
enmascarado = [2.0, −1e9, 0.5]   →   softmax = [0.82, 0.00, 0.18]
```

### El slot 2 depende del slot 1

Hay restricciones **entre** los dos slots que una máscara por slot no puede ver:
los dos no pueden cambiar al mismo pokémon, ni usar mega/z/dynamax/tera a la vez
(solo uno por turno). Por eso se elige primero el slot 1 y después el slot 2 con
`second_slot_mask(mask_slot2, a1)`:

- si `a1` es un cambio al pokémon k → se quita la acción `k` del slot 2;
- si `a1` usa un gimmick (por ejemplo mega: acciones 27–46) → se quitan las
  acciones del mismo gimmick del slot 2;
- si no queda nada legal, el slot 2 pasa (caso típico: dos pokémon debilitados
  y un solo reemplazo).

Sin esto, en la mayoría de las partidas de prueba aparecía al menos una
combinación ilegal. Con esto, 0 en 25 partidas. Si aun así una combinación fuera ilegal, el entorno usa
`strict=False`: juega el orden por defecto del servidor en vez de detener el
entrenamiento.

---

## 6. Reward: cómo se puntúa (`rewards/reward.py`)

Se reutiliza `reward_computing_helper` de poke-env. Calcula un **valor del
estado** y el reward de un turno es la **diferencia** con el valor del turno
anterior:

```
valor = Σ HP de los míos − Σ HP de los rivales           (× 1.0)
      − 2.0 × (míos debilitados) + 2.0 × (rivales debilitados)
      − 0.3 × (míos con estado)  + 0.3 × (rivales con estado)
      ± 30  al ganar / perder

reward del turno = valor ahora − valor antes
```

**Ejemplos:**

| Qué pasó en el turno | Reward |
|---|---|
| Le bajas 40% de HP a un rival | +0.4 |
| Debilitas a un rival que tenía 30% de HP | +0.3 (HP) + 2.0 (KO) = **+2.3** |
| Pierdes un pokémon que estaba al 100% | −1.0 − 2.0 = **−3.0** |
| Quemas a un rival | +0.3 |
| Ganas la partida | **+30** (en el último paso) |

**Por qué un reward "denso".** Con solo +30/−30 al final, la red tendría que
adivinar cuál de ~12 decisiones fue la buena. El HP y los KOs dan señal en cada
turno. Es *shaping*: puede sesgar hacia jugar "seguro" en lugar de ganar, así que
la victoria pesa mucho más (30) que cualquier turno suelto.

Un episodio típico suma entre ±32 y ±41. En `train.py` los rewards se dividen por
30 (`REWARD_SCALE`): ganar queda en ±1 y el error del crítico se mantiene en una
escala estable.

---

## 7. El entorno y el self-play (`envs/vgc_env.py`)

`VGCDoublesEnv` hereda de `DoublesEnv` de poke-env, que es un `ParallelEnv` de
PettingZoo con **dos agentes** (`agent1` y `agent2`) peleando en la misma
batalla del servidor local. Cada uno tiene su propia observación y su propio
reward, y ambos los controla quien use el entorno. Eso es lo que hace posible el
**self-play**: la misma red puede jugar los dos lados.

```python
env = VGCDoublesEnv()
obs, info = env.reset()                    # arranca una partida nueva
while env.agents:                          # env.agents queda vacío al terminar la partida
    actions = {a: env.random_action(a) for a in env.agents}
    obs, rewards, terminated, truncated, info = env.step(actions)
```

- `obs[agente]`: vector de 430 números de ese agente.
- `rewards[agente]`: el reward del turno para ese agente.
- `env.action_masks(agente)`: acciones legales `(2, 107)` de ese agente ahora.
- `env.random_action(agente)`: una jugada legal al azar (útil de rival o de prueba).

**Quién mueve en cada paso.** No siempre juegan los dos: si un pokémon se
debilita, solo el dueño debe elegir reemplazo mientras el otro espera. Los
atributos `env.agent1_to_move` y `env.agent2_to_move` dicen quién debe actuar en
este paso; el entrenamiento solo guarda decisiones de quien realmente movió.

**Equipos.** En el constructor: `agent1` recibe el equipo oficial y `agent2` un
`MetaPoolTeambuilder`. `agent1` es siempre el "oficial" y `agent2` un equipo meta
distinto cada partida.

**Velocidad.** Unos 77 pasos por segundo (una partida dura ~0.15 s), muchísimo más
rápido que jugar con los bots por defecto de poke-env.

---

## 8. La red actor-crítico (`agents/actor_critic.py`)

```
observación (430)
      │
   Linear 430→256, Tanh
   Linear 256→256, Tanh          ← tronco compartido
      │
      ├── cabeza slot 1: Linear 256→107  → 107 logits (uno por acción)
      ├── cabeza slot 2: Linear 256→107  → 107 logits
      └── cabeza valor:  Linear 256→1    → V(s)
```

231.383 parámetros en total.

- **Actor (política).** Las cabezas de los slots dan *logits*. Enmascarados y
  pasados por softmax son probabilidades, y de ahí se **muestrea** la acción
  (no se toma siempre la mejor: explorar es parte del aprendizaje).
- **Crítico (valor).** `V(s)` estima cuánto reward total espera el agente desde
  ese estado (con los rewards escalados, entre −1 y 1 aprox.). En una partida
  pareja debería estar cerca de 0; con dos rivales caídos, alto. Es la
  referencia para saber si una jugada salió mejor o peor de lo esperado.
- **Inicialización ortogonal.** Las cabezas de política arrancan con ganancia
  0.01: la red parte casi uniforme (explora de todo) en vez de con preferencias
  aleatorias fuertes.

### Muestreo de una decisión (`act`)

1. La observación pasa por la red y salen los logits de ambos slots y `V(s)`.
2. Se enmascaran los logits del slot 1 y se muestrea `a1`.
3. Con `second_slot_mask(máscara_slot2, a1)` se enmascara el slot 2 y se muestrea `a2`.
4. Se devuelve `[a1, a2]`, la máscara del slot 2 usada, el log-prob conjunto y `V(s)`.

**Log-prob conjunto.** `p(a1, a2) = p(a1) · p(a2 | a1)`, y con logaritmos los
productos son sumas:

```
p(a1) = 0.25 ,  p(a2 | a1) = 0.50
p(a1, a2) = 0.125        →   log p = ln 0.25 + ln 0.50 = −1.386 − 0.693 = −2.079
```

Se **guardan las máscaras usadas** porque en el update hay que recalcular
"¿qué probabilidad le da ahora la red a la acción que jugué entonces?", y esa
cuenta debe usar exactamente las mismas máscaras.

### Por qué la máscara vale −1e9 y no −∞

La entropía calcula `p · log p`. Con un logit `−∞` la probabilidad es 0 y queda
`0 · (−∞) = NaN`. Con `−1e9` (finito) da `0 · (−1e9) = 0`.

### `evaluate`

Recalcula, en lote y con la red **actual**, log-prob, entropía y valor de jugadas
ya hechas. Es lo que usa el update de PPO. Se comprobó que da los mismos
log-probs que `act` (diferencia menor a 1e-6).

---

## 9. PPO paso a paso (`agents/ppo.py`)

### La idea

El agente juega, y por cada jugada se pregunta: **¿salió mejor o peor de lo que
esperaba?** Si salió mejor, sube la probabilidad de esa jugada; si peor, la baja.
Lo difícil es (a) medir "mejor de lo esperado" cuando el reward llega al final y
(b) no cambiar la política tan bruscamente que se rompa.

### 9.1 Trayectorias y `Transition`

Cada decisión de un agente se guarda como una `Transition`:

```
obs | acción [a1,a2] | máscara slot 1 | máscara slot 2 | log-prob | V(s) | reward
```

El **reward** de una decisión es el acumulado hasta la siguiente decisión del
mismo agente (incluye rewards de pasos donde ese agente no movió, por ejemplo
cuando el rival le bajó HP mientras esperaba).

Las decisiones de un agente en una partida forman una **trayectoria**.

### 9.2 Ventaja: "mejor de lo esperado" (GAE)

La **ventaja** `A` de una decisión dice cuánto mejor (o peor) salió el resultado
que lo que el crítico esperaba.

Cada paso genera una *sorpresa*:

```
δ_t = r_t + γ · V(s_{t+1}) − V(s_t)
      └ lo que obtuve ┘        └ lo que esperaba ┘
```

Un solo `δ` solo ve un paso. **GAE** mezcla las sorpresas futuras con peso
decreciente para que el crédito de un final feliz viaje hacia atrás:

```
A_t = δ_t + (γλ) · δ_{t+1} + (γλ)² · δ_{t+2} + …          (γ = 0.99, λ = 0.95)
retorno_t = A_t + V(s_t)                                   (objetivo del crítico)
```

Se calcula hacia atrás con `A_t = δ_t + γλ · A_{t+1}`; tras la última decisión de
la partida el valor siguiente es 0 (la partida terminó).

**Ejemplo** (números crudos, sin escalar). Una partida de 3 decisiones donde solo
la última da reward (ganar, +30) y el crítico esperaba `V = [5, 12, 20]`:

```
δ₃ = 30 + 0        − 20 = 10          (no hay siguiente estado)
δ₂ =  0 + 0.99·20  − 12 = 7.8
δ₁ =  0 + 0.99·12  −  5 = 6.88

A₃ = 10
A₂ = 7.8  + 0.99·0.95 · 10     = 17.2
A₁ = 6.88 + 0.99·0.95 · 17.2   = 23.06

retornos = A + V = [28.06, 29.2, 30]
```

Las tres decisiones quedan con ventaja **positiva** porque el crítico subestimó
esta partida (creía 5, 12 y 20; ganar dio 30). Las más antiguas reciben más
crédito porque acumulan todo lo que vino después. `λ` controla cuánto se mira al
futuro: λ chico confía más en el crítico (menos ruido, más sesgo); λ grande
confía en lo que realmente pasó (más ruido, menos sesgo).

Antes del update, las ventajas se **normalizan** por lote (media 0, desviación 1).

### 9.3 La pérdida de política con recorte

Queremos subir la probabilidad de las jugadas con `A > 0` y bajar la de las de
`A < 0`. Pero un paso demasiado grande puede destrozar la política. PPO lo evita
comparando la probabilidad nueva con la vieja:

```
ratio = prob_nueva(jugada) / prob_vieja(jugada)
pérdida_política = − min( ratio · A ,  clip(ratio, 1−ε, 1+ε) · A )       (ε = 0.2)
```

**Ejemplos con ε = 0.2:**

| A | prob vieja → nueva | ratio | ¿Qué pasa? |
|---|---|---|---|
| +2 | 0.10 → 0.11 | 1.1 | Dentro de [0.8, 1.2]: el gradiente empuja a subirla más |
| +2 | 0.10 → 0.15 | 1.5 | `min(3.0, 1.2·2 = 2.4)` = 2.4, una constante: **gradiente 0**. "Ya la subiste lo suficiente" |
| −2 | 0.10 → 0.07 | 0.7 | `min(−1.4, 0.8·(−2) = −1.6)` = −1.6, constante: gradiente 0. "Ya la bajaste lo suficiente" |
| −2 | 0.10 → 0.13 | 1.3 | `min(−2.6, −2.4)` = −2.6: el gradiente **sí** actúa y la baja. El recorte nunca frena que se corrija un error |

Efecto: la política no se aleja mucho de la que generó los datos, y por eso se
pueden **reusar los mismos datos varias épocas** (4) en vez de descartarlos tras
un solo uso.

### 9.4 Pérdida total

```
pérdida = pérdida_política + 0.5 · pérdida_valor − 0.01 · entropía
```

- **pérdida_valor** = error cuadrático medio entre `V(s)` y el retorno. Un
  crítico mejor da ventajas más fiables.
- **entropía** = qué tan repartidas están las probabilidades. Se resta (se
  premia) para que la red no se vuelva determinista demasiado pronto y siga
  explorando. Es la suma de la entropía de los dos slots.

Se optimiza con Adam (lr 3e-4) y recorte de gradiente a norma 0.5.

**Ejemplo de entropía.** Elegir al azar entre 4 acciones legales da `ln 4 = 1.39`;
entre 14 da `ln 14 = 2.64`. Un valor conjunto de ~3.5 al inicio equivale a elegir
al azar entre ~6 opciones por slot.

### 9.5 Por qué PPO es "on-policy"

El `ratio` asume que las jugadas las generó una política muy parecida a la que se
está actualizando. Por eso solo se entrena con jugadas hechas por la red actual y
después se descartan. Es también por lo que el self-play puro (la misma red en
ambos lados) encaja tan bien: **todas** las jugadas de ambos agentes sirven. Si
el rival fuera una versión vieja de la red, sus jugadas habría que descartarlas.

### 9.6 Hiperparámetros (`python train.py --help`)

| Parámetro | Valor | Qué hace |
|---|---|---|
| `--steps` | 2048 | Decisiones a juntar antes de cada update (~100 partidas) |
| `--epochs` | 4 | Pasadas sobre esos datos |
| `--minibatch` | 256 | Tamaño de cada paso de gradiente |
| `--lr` | 3e-4 | Tasa de aprendizaje de Adam |
| `--gamma` | 0.99 | Descuento del futuro (las partidas son cortas, casi no descuenta) |
| `--lam` | 0.95 | λ de GAE (sesgo vs ruido) |
| `--clip` | 0.2 | ε del recorte: cuánto puede cambiar una probabilidad por update |
| `--ent-coef` | 0.01 | Peso del bono de entropía: más alto explora más |
| `--updates` | 200 | Cantidad de updates |
| `--eval-every` | 5 | Cada cuántos updates se evalúa y se guardan pesos |
| `--eval-episodes` | 100 | Partidas por evaluación |

---

## 10. El loop de entrenamiento (`train.py`)

```
repetir N updates:
    1. RECOLECTAR   jugar partidas completas en self-play (misma red en los dos lados)
                    hasta juntar ≥ 2048 decisiones
    2. VENTAJAS     por cada trayectoria: GAE → ventajas y retornos
    3. ACTUALIZAR   PPO: 4 épocas sobre esos datos (minibatches de 256)
    4. EVALUAR      (cada 5 updates) win rate contra rival al azar, sobre 100 partidas
    5. GUARDAR      checkpoints/latest.pt siempre; best.pt si mejora el win rate
```

**Recolección (`collect_rollout`).** Por cada partida: `env.reset()`; mientras haya
agentes, para cada uno que deba mover se calcula su máscara, `act` elige la jugada y
se guarda una `Transition`; luego `env.step(...)` y el reward del paso se suma a la
última decisión de cada agente (dividido por 30). Al terminar la partida cada
trayectoria pasa por GAE y se agrega al buffer. Se juegan partidas enteras (no se
corta a la mitad), así el final de cada trayectoria es un final real.

**Entrenan las jugadas de los dos agentes**, así la red aprende a jugar también los
equipos meta, no solo el oficial.

**Evaluación.** El agente 1 (equipo oficial) juega de forma determinista (siempre
la jugada más probable) contra un rival que elige jugadas legales al azar. Se
imprime también la **referencia**: cuánto gana un agente al azar contra otro al
azar (~50%; es la ventaja del equipo en sí). Una red *sin entrenar* rindió entre 16% y 51% en las 4 semillas probadas.

### Qué mirar (TensorBoard: `tensorboard --logdir runs`)

| Métrica | Cómo leerla |
|---|---|
| `eval/win_rate_vs_random` | El objetivo, pero **satura rápido**: tras 2 updates ya ganaba ~89%. Ver sección 12 |
| `selfplay/mean_return_oficial` | Reward medio del agente oficial en self-play; ronda 0 si ambos lados van parejos |
| `train/entropy` | Debe bajar despacio. Si cae a ~0 de golpe, la política se volvió determinista demasiado pronto |
| `train/approx_kl` | Cuánto se movió la política en el update. Normal < ~0.02; si se dispara, baja el `lr` |
| `train/clip_fraction` | Fracción de jugadas recortadas (normal 0.05–0.3). Cerca de 0: updates muy tímidos |
| `train/value_loss` | Error del crítico; debería bajar |
| `train/explained_variance` | Qué parte del retorno explica el crítico (1 = perfecto; ≤ 0 = no aporta) |
| `train/policy_loss` | Poco informativa por sí sola |

### Guardado de pesos (`agents/checkpoint.py`)

`checkpoints/latest.pt` (último) y `checkpoints/best.pt` (mejor win rate de
evaluación). Cada archivo guarda pesos, estado del optimizador (para retomar), el
número de update y el tamaño de la observación y de las acciones con que se
entrenó. `load_model` rechaza un checkpoint incompatible con el código actual
(por ejemplo, si cambió el tamaño de la observación) con un mensaje claro.

---

## 11. Jugar contra la IA (`play.py`)

`play.py` levanta un bot (`TrainedPlayer`) que carga los pesos de `checkpoints/` y
**acepta desafíos** en el servidor local. Tú juegas desde el navegador.

1. Servidor corriendo y `python play.py` (usa `checkpoints/pretrained.pt` por defecto).
2. Abre `http://localhost:8000` y elige un nombre cualquiera (sin contraseña).
3. Arma o importa un equipo válido para el formato: sirve cualquiera de
   `teams/data/meta/*.txt` (Teambuilder → Import).
4. En el chat escribe `/challenge PokeEmoBot, gen9championsvgc2026regmc` (se usa el equipo que
   tengas seleccionado para ese formato).

La IA juega con el equipo oficial. En cada turno `TrainedPlayer.choose_move`:
`embed_battle` → `legal_action_masks` → `model.act` → `DoublesEnv.action_to_order`
(el par de acciones pasa a una orden que entiende el servidor).

Opciones: `--checkpoint checkpoints/latest.pt`, `--name`, `--battles N` y
`--stochastic` (la IA muestrea sus jugadas en vez de elegir siempre la más
probable: juega más variado).

---

## 12. Limitaciones y próximos pasos

Lo que este sistema **no** hace todavía, para no olvidarlo:

- **La evaluación contra un rival al azar se satura rápido.** Con 2 updates ya
  gana ~89%, así que deja de servir para medir progreso. Conviene evaluar contra
  versiones anteriores de la red (un pequeño "campeonato") o contra un bot con
  heurística (por ejemplo, "elige el movimiento de más daño esperado").
- **Team preview aleatorio.** poke-env elige al azar qué 4 de los 6 pokémon
  llevar. La red no aprende a elegirlos.
- **Sin memoria.** El MLP ve solo el estado actual. No recuerda los movimientos ni
  los objetos que el rival ya reveló. Una red recurrente (LSTM) o agregar esos
  datos a la observación lo mejoraría.
- **Rival = self-play puro con la misma red.** Puede ciclar (aprender a ganarle a su
  propia versión actual y olvidar cómo ganar a otras). La mitigación típica es
  jugar a veces contra versiones antiguas guardadas, entrenando solo con el agente
  que usa la red actual (por ser PPO on-policy).
- **Reward con shaping.** El HP y los KOs pueden sesgar hacia jugar "seguro". Si el
  agente se comporta de forma rara, revisar los pesos de `rewards/reward.py`.
- **Solo mega evolución.** Las acciones de z-move, dynamax y tera existen en el
  espacio de acciones pero este formato no las permite: siempre están enmascaradas.
- **Los meta 3 y 5 son prácticamente el mismo equipo** (mismas especies, sets y
  movimientos): en la práctica hay 4 rivales distintos.
- **Un solo entorno.** Con unos 77 pasos/s alcanza para empezar; para escalar se
  pueden correr varios entornos en paralelo (cada uno con su propio par de
  usuarios).

---

## 13. Problemas frecuentes

| Síntoma | Causa | Solución |
|---|---|---|
| Error `nametaken` con "authentication token was invalid" | El servidor no acepta nombres sin login | `noguestsecurity = true` en `config.js` y reiniciar |
| La 2ª partida seguida se cuelga / `Agent is not challenging` | El servidor cancela desafíos con menos de 10 s de diferencia | `nothrottle = true` en `config.js` y reiniciar |
| La batalla nunca empieza (sin error) | El servidor rechazó el equipo (por ejemplo EVs en vez de Stat Points, o un movimiento ilegal) | `./pokemon-showdown validate-team gen9championsvgc2026regmc < equipo.txt` |
| `KeyError: 'froslassmega'` (o similar) | El nombre del equipo trae la forma mega, o falta la forma en el pokedex de poke-env | Escribir la especie base + mega stone; el parche de `teams/` cubre las formas mega |
| `The Pokemon "basculegionmale" does not exist` | Nombre de especie inválido | Usar `Basculegion` (el macho es la especie base) |
| `ValueError: ... hay que reentrenar` al cargar | El checkpoint se entrenó con otro tamaño de observación/acciones | Reentrenar, o volver al código con que se entrenó |
| `No existe checkpoints/best.pt` | Aún no se ha entrenado | `python train.py` |

---

## 14. Glosario

- **Agente**: el jugador controlado por la red.
- **Política (π)**: la regla que asigna a cada estado una distribución de probabilidad sobre acciones. Aquí la produce la red (el "actor").
- **Valor `V(s)`**: reward total que se espera recibir desde un estado. Lo estima el "crítico".
- **Logit**: el puntaje bruto de una acción, antes del softmax.
- **Softmax**: convierte logits en probabilidades que suman 1.
- **Trayectoria**: la secuencia de decisiones de un agente en una partida.
- **Retorno**: suma (descontada) de los rewards futuros desde una decisión.
- **Ventaja**: cuánto mejor (o peor) salió una jugada de lo que el crítico esperaba.
- **GAE**: forma de estimar la ventaja mezclando sorpresas de varios pasos.
- **On-policy**: se entrena solo con datos generados por la política actual.
- **Self-play**: el agente juega contra copias de sí mismo.
- **Entropía**: qué tan repartidas están las probabilidades (alta = explora, ~0 = determinista).
- **KL aproximada**: cuánto cambió la política respecto a la que generó los datos.
- **Máscara de acciones**: marca cuáles acciones son legales; las ilegales tienen probabilidad 0.
- **Stat Points (SP)**: reemplazo de los EVs en este formato (máx. 32 por stat, 66 en total).
- **Shaping**: rewards intermedios (HP, KOs) que guían el aprendizaje además del resultado final.
