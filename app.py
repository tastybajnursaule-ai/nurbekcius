"""AI Learning Hub — Есептер + Курстар + Gemini AI бір жүйеде.
Іске қосу: streamlit run app.py
"""
import base64
import json
import os
import random
import re

import numpy as np
import pandas as pd
import streamlit as st
from sympy import (Poly, Rational, diff, expand, factor, integrate, nsimplify,
                   simplify, solve, sqrt, symbols, sympify, lambdify)
from sympy.parsing.sympy_parser import (convert_xor, implicit_multiplication_application,
                                        parse_expr, standard_transformations)

x, y = symbols("x y")
TR = standard_transformations + (implicit_multiplication_application, convert_xor)


# ===================== МАТЕМАТИКА =====================
def norm(t):
    return t.replace("²", "^2").replace("³", "^3").replace("−", "-").replace("×", "*")


def P(s):
    e = safe_expr(s)
    if e is None:
        raise ValueError("Рұқсат етілмеген таңба немесе функция")
    return parse_expr(e.replace(",", "."), transformations=TR)


def solve_equation(expr):
    p = Poly(expand(expr), x)
    deg, steps, roots = p.degree(), [], []
    if deg == 1:
        a, b = p.all_coeffs()
        steps = [f"Әдіс: белгісізді бір жаққа шығару. {a}x + ({b}) = 0", f"x = {-b}/{a} = {Rational(-b, a)}"]
        roots = [Rational(-b, a)]
        kind = "Сызықтық теңдеу"
    elif deg == 2:
        a, b, c = p.all_coeffs()
        D = b**2 - 4*a*c
        steps = ["Әдіс: дискриминант. D = b² − 4ac", f"a = {a}, b = {b}, c = {c}", f"D = {D}"]
        if D > 0:
            roots = [((-b + sqrt(D)) / (2*a)).simplify(), ((-b - sqrt(D)) / (2*a)).simplify()]
            steps += ["D > 0: екі түбір, x = (−b ± √D) / (2a)", f"x₁ = {roots[0]}, x₂ = {roots[1]}"]
        elif D == 0:
            roots = [Rational(-b, 2*a)]
            steps += ["D = 0: бір түбір, x = −b / (2a)", f"x = {roots[0]}"]
        else:
            steps += ["D < 0: нақты түбір жоқ."]
        kind = "Квадрат теңдеу"
    else:
        roots = [r for r in solve(expr, x) if r.is_real]
        steps = ["Әдіс: көбейткіштерге жіктеу / сандық шешу (SymPy)", f"Нақты түбірлер: {roots}"]
        kind = f"{deg}-дәрежелі теңдеу"
    ans = ", ".join(f"x{i+1} = {r}" for i, r in enumerate(roots)) or "нақты түбір жоқ"
    return kind, steps, ans, roots


def solve_any(text, mode):
    t = norm(text.strip())
    low = t.lower()
    if mode == "Авто":
        mode = ("Жүйе" if ";" in t else "Теңдеу" if "=" in t else
                "Туынды" if low.startswith(("diff", "туынды")) else
                "Интеграл" if low.startswith(("int", "интеграл", "∫")) else "Жіктеу")
    body = re.sub(r"^(diff|туынды|integral|int|интеграл|∫)\s*", "", t, flags=re.I).replace("dx", "")
    if mode == "Теңдеу":
        l, r = t.split("=", 1)
        return (mode,) + solve_equation(P(l) - P(r))
    if mode == "Жүйе":
        eqs = [P(a) - P(b) for a, b in (e.split("=", 1) for e in t.split(";"))]
        sol = solve(eqs, [x, y], dict=True)
        ans = ", ".join(f"{k} = {v}" for k, v in sol[0].items()) if sol else "шешімі жоқ"
        return mode, "Теңдеулер жүйесі", ["Әдіс: алмастыру / қосу (SymPy)", f"Жауап: {ans}"], ans, (sol[0] if sol else None)
    if mode == "Туынды":
        f = P(body)
        return mode, "Туынды табу", [f"f(x) = {f}", "Дәреже мен көбейтінді ережелерін қолданамыз", f"f'(x) = {diff(f, x)}"], f"f'(x) = {diff(f, x)}", diff(f, x)
    if mode == "Интеграл":
        f = P(body)
        F = integrate(f, x)
        return mode, "Анықталмаған интеграл", [f"∫ {f} dx", f"= {F} + C"], f"{F} + C", F
    f = P(t)
    return "Жіктеу", "Көпмүшелікті жіктеу", [f"Өрнек: {f}", f"Көбейткіштерге жіктеу: {factor(f)}"], f"{factor(f)}", factor(f)


def check_answer(mode, truth, user_text):
    try:
        if mode == "Жүйе":
            vals = {k: nsimplify(sympify(v)) for k, v in re.findall(r"([xy])\s*=\s*(-?[\d./]+)", norm(user_text))}
            good = {str(k): nsimplify(v) for k, v in truth.items()}
            return (True, "Дұрыс!") if vals == good else (False, "x және y мәндерін қайта тексеріңіз. Үлгі: x = 3, y = 2")
        if mode == "Теңдеу":
            t = re.sub(r"x\s*\d*\s*=", "", norm(user_text).replace(",", " "))
            u = {nsimplify(sympify(v)) for v in re.split(r"[;\s]+", t.strip()) if v}
            tr = {nsimplify(v) for v in truth}
            if u == tr:
                return True, "Дұрыс!"
            return False, ("Түбірлер жетіспейді." if u < tr else
                           "Таңбаны, 2a-ға бөлуді және D есебін тексеріңіз. Түбірді теңдеуге қойып көріңіз.")
        u = P(user_text.split("=")[-1].replace("+ C", "").replace("+C", ""))
        return (True, "Дұрыс!") if simplify(u - truth) == 0 else (False, "Жауап сәйкес емес. Қадамдарды қайта қарап шығыңыз.")
    except Exception:
        return False, "Жауабыңызды оқи алмадым."


def new_problem(mode):
    a, b = random.randint(2, 7), random.randint(1, 9)
    if mode == "Жүйе":
        return f"x + y = {a+b}; x - y = {a-b}"
    if mode == "Туынды":
        return f"туынды {a}x^3 + {b}x"
    if mode == "Интеграл":
        return f"интеграл {a}x^2 + {b}"
    if mode == "Жіктеу":
        return f"x^2 + {a+b}x + {a*b}"
    r1, r2, k = random.randint(-6, 6), random.randint(-6, 6), random.choice([1, 2, 3])
    c = Poly(expand(k * (x - r1) * (x - r2)), x).all_coeffs()
    return f"{c[0]}x^2 + ({c[1]})x + ({c[2]}) = 0"


# ===================== AI ПРОВАЙДЕРЛЕР =====================
ENV = {"Gemini": "GEMINI_API_KEY"}
MODELS = {"Gemini": "gemini-2.5-flash"}
SYSTEM = "Сен қазақ тілінде сөйлейтін математика және Python мұғалімісің. Қысқа, түсінікті, қадамдап жауап бер."


def get_key(name):
    k = st.session_state.get("key_" + name) or os.getenv(ENV[name])
    if not k:
        try:
            k = st.secrets.get(ENV[name])
        except Exception:
            k = None
    return k


def ask_ai(name, prompt):
    key = get_key(name)
    if not key:
        return f"⚠️ {name} үшін API кілті жоқ. Сол жақ мәзірден қосыңыз."
    model = st.session_state.get("model_" + name) or MODELS[name]
    try:
        from google import genai
        return genai.Client(api_key=key).models.generate_content(
            model=model, contents=SYSTEM + "\n\n" + prompt).text
    except Exception as e:
        return f"⚠️ {name} қатесі: {e}"


def transcribe_image(name, data, mime):
    """Суреттегі есептің шартын AI көмегімен мәтінге айналдырады."""
    key = get_key(name)
    if not key:
        return None
    model = st.session_state.get("model_" + name) or MODELS[name]
    ask = ("Суреттегі есептің шартын дәл көшіріп жаз. Формулаларды x^2, sqrt(3), 1/2 сияқты "
           "қарапайым мәтінмен жаз. Тек есептің мәтінін қайтар, шешпе.")
    try:
        b64 = base64.b64encode(data).decode()
        from google import genai
        from google.genai import types
        return genai.Client(api_key=key).models.generate_content(
            model=model, contents=[types.Part.from_bytes(data=data, mime_type=mime), ask]).text
    except Exception as e:
        return f"⚠️ {name} суретті оқи алмады: {e}"


# ===================== УНИВЕРСАЛ ШЕШУШІ (AI + SymPy тексеруі) =====================
SAFE = {"sqrt", "sin", "cos", "tan", "log", "ln", "exp", "pi", "E", "Abs"}
JSON_PROMPT = (
    "Төмендегі есепті шеш. Жауапты ТЕК JSON түрінде қайтар (markdown жоқ):\n"
    '{"type": "есеп түрі", "steps": ["1-қадам", "2-қадам"], "answer": "қысқа соңғы жауап қазақша", '
    '"equation": "есепті x айнымалысымен бір теңдеуге келтірсең, мысалы 2x^2-5x+2=0, болмаса бос жол", '
    '"answer_value": "сандық жауап өрнек түрінде, мысалы 2; 1/2 (бірнешеу болса ; арқылы, ондық бөлшек нүктемен), болмаса бос жол"}\n'
    "Есеп:\n")


def safe_expr(t):
    t = norm(t or "").strip()
    if not t or "__" in t or not re.fullmatch(r"[0-9A-Za-z\s+\-*/^().,_=]+", t):
        return None
    if any(len(n) > 1 and n not in SAFE for n in re.findall(r"[A-Za-z_]+", t)):
        return None
    return t


def parse_values(text):
    out = set()
    for v in (text or "").split(";"):
        e = safe_expr(v)
        if e:
            try:
                out.add(nsimplify(P(e)))
            except Exception:
                pass
    return out


def same(a, b):
    try:
        fa, fb = sorted(float(v) for v in a), sorted(float(v) for v in b)
        return len(fa) == len(fb) and all(abs(p - q) < 1e-6 for p, q in zip(fa, fb))
    except Exception:
        return a == b


def verify(d):
    ai = parse_values(d.get("answer_value", ""))
    eq = safe_expr(d.get("equation", ""))
    if eq and "=" in eq:
        try:
            l, r = eq.split("=", 1)
            sol = {nsimplify(v) for v in solve(P(l) - P(r), x) if v.is_real}
            txt = ", ".join(map(str, sol)) or "нақты түбір жоқ"
            if ai and same(ai, sol):
                return f"✅ SymPy өз бетінше тексерді: жауап сәйкес ({txt})"
            return f"⚠️ SymPy басқа нәтиже берді: {txt}. Шешуді мұқият тексеріңіз."
        except Exception:
            pass
    return "ℹ️ Автоматты тексеру мүмкін емес (мәтіндік немесе дәлелдеу есебі). Қадамдарды өзіңіз қарап шығыңыз."


def solve_with_ai(name, prob):
    raw = ask_ai(name, JSON_PROMPT + prob)
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(m.group(0))
    except Exception:
        return {"raw": raw}


# ===================== КУРСТАР =====================
COURSES = {
    "Python негіздері": [
        {"t": "Айнымалылар мен print", "x": "Айнымалы — деректі сақтайтын атау. print() нәтижені экранға шығарады.",
         "c": "name = 'Айдар'\nage = 15\nprint(name, age)",
         "q": ("age = 15 болса, age + 1 неге тең?", ["14", "16", "151"], 1)},
        {"t": "Цикл және шарт", "x": "if шартты тексереді, for әрекетті қайталайды.",
         "c": "for i in range(1, 4):\n    if i % 2 == 0:\n        print(i, 'жұп')",
         "q": ("range(1, 4) қандай сандарды береді?", ["1, 2, 3", "1, 2, 3, 4", "0, 1, 2, 3"], 0)},
        {"t": "Тізімдер мен функциялар", "x": "Тізім — бірнеше мәнді сақтайды, функция — қайталанатын жұмысты бір атпен жинайды.",
         "c": "def square(n):\n    return n * n\n\nnums = [1, 2, 3]\nprint([square(n) for n in nums])",
         "q": ("square(4) нәтижесі қандай?", ["8", "16", "44"], 1)}],
    "Жасанды интеллект негіздері": [
        {"t": "ЖИ және ML", "x": "Machine Learning ережені адам жазбай, деректен үйренеді. Негізгі қадамдар: деректер, модель, оқыту, тестілеу.",
         "c": "from sklearn.linear_model import LinearRegression\nm = LinearRegression().fit([[1],[2],[3]], [2,4,6])\nprint(m.predict([[4]]))",
         "q": ("Модель қай кезеңде мысалдардан үйренеді?", ["Тестілеу", "Оқыту", "Деректі жинау"], 1)},
        {"t": "ЖИ түрлері", "x": "Gemini, ChatGPT, Claude сияқты жүйелер — үлкен тіл модельдері (LLM). Олар мәтінді түсініп, жауап жасайды. Бұл жобада Gemini қолданылады.",
         "c": "# Әр провайдердің API-і арқылы бір сұрақ жіберуге болады\nanswer = ask_ai('Gemini', 'Дискриминант деген не?')",
         "q": ("LLM нені жақсы істейді?", ["Мәтінмен жұмыс", "Суретті басып шығару", "Интернет жылдамдату"], 0)},
        {"t": "Нейрон және оқыту", "x": "Нейрон кірісті салмаққа көбейтіп қосады. Оқыту дегеніміз — қатені азайту үшін салмақты аздап өзгерте беру (градиенттік түсу). Мұны «🔬 AI зертханасында» көре аласыз.",
         "c": "w, b = 0.0, 0.0\nfor _ in range(100):\n    err = (w * 3 + b) - 7\n    w -= 0.01 * 2 * err * 3\n    b -= 0.01 * 2 * err",
         "q": ("Оқыту кезінде модель нені азайтады?", ["Қатені", "Деректер санын", "Файл көлемін"], 0)}],
    "Алгебра": [
        {"t": "Квадрат теңдеу", "x": "ax² + bx + c = 0 теңдеуінде D = b² − 4ac. D > 0 болса екі түбір бар.",
         "c": "from sympy import symbols, solve\nx = symbols('x')\nprint(solve(2*x**2 - 5*x + 2, x))",
         "q": ("x² − 5x + 6 = 0 үшін D неге тең?", ["1", "25", "49"], 0)},
        {"t": "Туынды", "x": "Туынды — функцияның өзгеру жылдамдығы. (xⁿ)' = n·xⁿ⁻¹.",
         "c": "from sympy import symbols, diff\nx = symbols('x')\nprint(diff(x**3, x))",
         "q": ("(x³)' неге тең?", ["3x²", "x²", "3x"], 0)},
        {"t": "Теңдеулер жүйесі", "x": "Екі белгісізі бар екі теңдеуді алмастыру немесе қосу әдісімен шешеміз. Мысалы: x + y = 5 және x − y = 1 болса, қосқанда 2x = 6, яғни x = 3, y = 2.",
         "c": "from sympy import symbols, solve\nx, y = symbols('x y')\nprint(solve([x + y - 5, x - y - 1], [x, y]))",
         "q": ("x + y = 5 және x − y = 1 үшін x неге тең?", ["2", "3", "4"], 1)}],
}


# ===================== ИНТЕРФЕЙС =====================
st.set_page_config(page_title="AI Learning Hub", page_icon="🧠", layout="wide")
ss = st.session_state
ss.setdefault("done", set())
ss.setdefault("chat", [])
ss.setdefault("xp", 0)
ss.setdefault("ok", 0)
ss.setdefault("n", 0)

with st.sidebar:
    st.title("🧠 AI Learning Hub")
    page = st.radio("Бөлім", ["🏠 Басты бет", "🧮 Есептер", "🧠 Кез келген есеп", "🎯 Жаттығу", "📈 График", "🔬 AI зертханасы", "📚 Курстар", "🤖 AI чат"])
    with st.expander("🔑 AI кілттері"):
        st.caption("Кілттер тек осы сессияда сақталады. Оларды GitHub-қа жүктемеңіз.")
        for n in ENV:
            ss["key_" + n] = st.text_input(f"{n} API key", type="password", key=f"k_{n}")
            ss["model_" + n] = st.text_input(f"{n} моделі", MODELS[n], key=f"m_{n}")
    total = sum(len(v) for v in COURSES.values())
    st.progress(len(ss.done) / total, f"Курс прогресі: {len(ss.done)}/{total}")

# ---------- Есептер ----------
# ---------- Басты бет ----------
if page == "🏠 Басты бет":
    st.title("🧠 AI Learning Hub")
    st.caption("Математика мен жасанды интеллект бір платформада")
    c = st.columns(4)
    c[0].metric("Деңгей", ss.xp // 100 + 1)
    c[1].metric("XP", ss.xp)
    c[2].metric("Сабақтар", f"{len(ss.done)}/{total}")
    c[3].metric("Жаттығу", f"{ss.ok}/{ss.n}")
    st.progress(min(ss.xp % 100, 100) / 100, "Келесі деңгейге дейін")
    if not get_key("Gemini"):
        st.info("AI мүмкіндіктері үшін сол жақтағы 🔑 бөлімге Gemini кілтін қосыңыз. Есептер, жаттығу, график, зертхана және курстар кілтсіз де жұмыс істейді.")
    st.subheader("Платформада не бар")
    st.markdown(
        "- 🧮 **Есептер**: теңдеу, жүйе, туынды, интеграл, жіктеу, қадамдап шешу\\n"
        "- 🧠 **Кез келген есеп**: мәтіндік және қиын есептер, суреттен оқу, SymPy тексеруі\\n"
        "- 🎯 **Жаттығу**: кездейсоқ есептер, жауап тексеру, XP жинау\\n"
        "- 📈 **График**: функция мен туындысының графигі\\n"
        "- 🔬 **AI зертханасы**: модель қалай үйренетінін көру\\n"
        "- 📚 **Курстар**: Python, ЖИ және алгебра сабақтары\\n"
        "- 🤖 **AI чат**: Gemini тьюторы")

elif page == "🧮 Есептер":
    st.header("🧮 Математикалық есептер")
    modes = ["Авто", "Теңдеу", "Жүйе", "Туынды", "Интеграл", "Жіктеу"]
    c1, c2 = st.columns([1, 3])
    mode = c1.selectbox("Есеп түрі", modes)
    ss.setdefault("problem", "2x^2 - 5x + 2 = 0")
    problem = c2.text_input("Есепті енгізіңіз", key="problem",
                            help="Жүйе: x + y = 5; x - y = 1. Туынды: туынды x^3. Интеграл: интеграл x^2")
    try:
        m, kind, steps, ans, truth = solve_any(problem, mode)
    except Exception as e:
        st.error(f"Есепті түсінбедім: {e}")
        st.stop()
    st.subheader(f"Түрі: {kind}")
    for i, s in enumerate(steps, 1):
        st.write(f"**{i}.** {s}")
    st.success(f"Жауабы: {ans}")

    prov = st.selectbox("AI түсіндірсін", list(ENV))
    if st.button("Түсіндір"):
        st.info(ask_ai(prov, f"Есеп: {problem}\nШешу қадамдары:\n" + "\n".join(map(str, steps)) +
                       "\nОсыны 8-сынып оқушысына түсіндір."))
    st.divider()
    a = st.text_input("Менің жауабымды тексер", placeholder="Мысалы: 2; 0.5")
    if st.button("Тексеру") and a and truth is not None:
        ok, msg = check_answer(m, truth, a)
        (st.success if ok else st.error)(msg)
    if st.button("Осыған ұқсас есеп бер"):
        ss.problem = new_problem(m if m != "Теңдеу" else "Теңдеу")
        st.rerun()
    up = st.file_uploader("Сурет арқылы енгізу (міндетті емес)", type=["png", "jpg", "jpeg"])
    if up:
        try:
            import cv2, numpy as np, pytesseract
            img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
            st.code(pytesseract.image_to_string(img, config="--psm 7").strip() or "мәтін табылмады")
        except ImportError:
            st.warning("Сурет режимі үшін opencv-python, pytesseract және Tesseract орнатыңыз.")

# ---------- Кез келген есеп ----------
elif page == "🧠 Кез келген есеп":
    st.header("🧠 Кез келген есеп")
    st.caption("Мәтіндік, геометрия, ықтималдық, олимпиада есептері: AI түсініп шешеді, SymPy нәтижені өз бетінше тексереді.")
    up = st.file_uploader("📷 Немесе есептің суретін жүктеңіз", type=["png", "jpg", "jpeg", "webp"], key="uimg")
    if up:
        st.image(up, width=320)
        vis = st.selectbox("Суретті қай AI оқысын?", list(ENV), key="vis")
        if st.button("Суреттен есепті оқу"):
            if not get_key(vis):
                st.warning(f"{vis} үшін API кілті жоқ. Сол жақтағы 🔑 бөліміне қосыңыз немесе басқа AI таңдаңыз.")
            else:
                with st.spinner("Сурет оқылып жатыр..."):
                    txt = transcribe_image(vis, up.getvalue(), up.type or "image/png")
                if txt and not txt.startswith("⚠️"):
                    ss.uprob = txt.strip()
                else:
                    st.error(txt or "Оқу мүмкін болмады.")
    prob = st.text_area("Есептің шартын жазыңыз (суреттен оқылған мәтінді түзетуге болады)", height=140, key="uprob")
    c1, c2 = st.columns(2)
    who = "Gemini"
    c1.caption("AI: Gemini")
    review = False
    if st.button("Шеш", type="primary") and prob.strip():
        names = list(ENV) if who.startswith("Үшеуі") else [who]
        if not any(get_key(n) for n in names):
            st.warning("AI кілті жоқ, сондықтан SymPy-мен жергілікті шешіп көремін.")
            try:
                m, kind, steps, ans, _ = solve_any(prob, "Авто")
                st.subheader(kind)
                for i, t in enumerate(steps, 1):
                    st.write(f"**{i}.** {t}")
                st.success(f"Жауабы: {ans}")
            except Exception:
                st.error("Бұл есепті кілтсіз шеше алмаймын. Сол жақтағы 🔑 бөліміне AI кілтін қосыңыз.")
        else:
            res = {}
            for n in names:
                if get_key(n):
                    with st.spinner(f"{n} шешіп жатыр..."):
                        res[n] = solve_with_ai(n, prob)
            cols = st.columns(len(res))
            for col, (n, d) in zip(cols, res.items()):
                with col:
                    st.subheader(n)
                    if "raw" in d:
                        st.write(d["raw"])
                        continue
                    st.caption(d.get("type", ""))
                    for i, t in enumerate(d.get("steps", []), 1):
                        st.write(f"**{i}.** {t}")
                    st.success(f"Жауабы: {d.get('answer', '')}")
                    st.write(verify(d))
            vals = [parse_values(d.get("answer_value", "")) for d in res.values() if "raw" not in d]
            vals = [v for v in vals if v]
            if len(vals) > 1:
                if all(same(vals[0], v) for v in vals[1:]):
                    st.success("✅ Барлық AI бірдей жауап берді.")
                else:
                    st.warning("⚠️ AI жауаптары әртүрлі. Қайсысы дұрыс екенін шешу жолынан тексеріңіз.")
            if review:
                first = next(iter(res))
                other = next((n for n in ENV if n != first and get_key(n)), None)
                if other and "raw" not in res[first]:
                    with st.spinner(f"{other} тексеріп жатыр..."):
                        st.info(f"🔍 {other} рецензиясы:\n\n" + ask_ai(other,
                            f"Есеп: {prob}\nШешу: {res[first].get('steps')}\nЖауап: {res[first].get('answer')}\n"
                            "Шешуде қате бар ма? Қате болса қайсы қадамда екенін көрсет, дұрыс болса қысқа растап бер."))

# ---------- Жаттығу ----------
elif page == "🎯 Жаттығу":
    st.header("🎯 Жаттығу")
    topic = st.selectbox("Тақырып", ["Теңдеу", "Жүйе", "Туынды", "Интеграл", "Жіктеу"])
    if st.button("Жаңа есеп") or "pp" not in ss or ss.get("pt") != topic:
        ss.pp, ss.pt, ss.pdone, ss.pcnt = new_problem(topic), topic, False, False
    st.subheader(ss.pp)
    st.caption("Жауап үлгісі: теңдеу 2; 0.5 | жүйе x = 3, y = 2 | туынды 3x^2 | жіктеу (x+4)^2")
    ans = st.text_input("Жауабыңыз", key="pa" + ss.pp)
    m, kind, steps, sol, truth = solve_any(ss.pp, "Авто")
    if st.button("Тексеру") and ans:
        ok, msg = check_answer(m, truth, ans)
        if not ss.pcnt:
            ss.n, ss.pcnt = ss.n + 1, True
        if ok:
            if not ss.pdone:
                ss.xp, ss.ok, ss.pdone = ss.xp + 10, ss.ok + 1, True
            st.success("Дұрыс! +10 XP")
        else:
            st.error(msg)
    with st.expander("Шешуін көру"):
        for i, t in enumerate(steps, 1):
            st.write(f"**{i}.** {t}")
        st.write(f"Жауабы: **{sol}**")

# ---------- График ----------
elif page == "📈 График":
    st.header("📈 Функция графигі")
    c1, c2, c3 = st.columns([3, 1, 1])
    txt = c1.text_input("f(x) =", "x^2 - 5x + 2")
    lo = c2.number_input("x min", value=-5.0)
    hi = c3.number_input("x max", value=8.0)
    try:
        f = P(txt)
        xs = np.linspace(lo, hi, 400)
        d = pd.DataFrame({"x": xs, "f(x)": lambdify(x, f, "numpy")(xs) * np.ones_like(xs)})
        if st.checkbox("Туындыны да көрсету"):
            d["f'(x)"] = lambdify(x, diff(f, x), "numpy")(xs) * np.ones_like(xs)
        st.line_chart(d, x="x", y=[c for c in d.columns if c != "x"])
        st.write(f"Туынды: **{diff(f, x)}**")
        roots = [r for r in solve(f, x) if r.is_real]
        st.write("Түбірлер (f(x) = 0): " + (", ".join(map(str, roots)) or "нақты түбір жоқ"))
    except Exception as e:
        st.error(f"Графикті салу мүмкін болмады: {e}")

# ---------- AI зертханасы ----------
elif page == "🔬 AI зертханасы":
    st.header("🔬 AI зертханасы: модель қалай үйренеді")
    st.write("Нүктелер y = 2x + 1 заңдылығына жақын. Модель w және b мәндерін алғашында білмейді, қатені азайта отырып тауып алады (градиенттік түсу).")
    c1, c2, c3 = st.columns(3)
    noise = c1.slider("Деректегі шу", 0.0, 5.0, 1.0)
    steps_n = c2.slider("Оқыту қадамы", 1, 500, 100)
    lr = c3.slider("Оқыту жылдамдығы", 0.001, 0.05, 0.01, format="%.3f")
    X = np.linspace(0, 10, 40)
    Y = 2 * X + 1 + np.random.default_rng(1).normal(0, noise, 40)
    w = b = 0.0
    losses = []
    for _ in range(steps_n):
        err = w * X + b - Y
        losses.append(float((err ** 2).mean()))
        w, b = w - lr * (2 * err * X).mean(), b - lr * (2 * err).mean()
    if not np.isfinite(w) or losses[-1] > 1e6:
        st.error("Қате өсіп кетті: оқыту жылдамдығы тым үлкен. Оны азайтып көріңіз.")
    else:
        st.success(f"Модель тапқан мән: w = {w:.2f}, b = {b:.2f} (нақты мән: 2 және 1). Қате: {losses[-1]:.2f}")
        st.scatter_chart(pd.DataFrame({"x": X, "деректер": Y, "модель": w * X + b}), x="x", y=["деректер", "модель"])
        st.caption("Қате (loss) қадам сайын қалай азаятыны:")
        st.line_chart(pd.DataFrame({"қате": losses}))
        if st.button("Gemini-ден түсіндіруін сұра"):
            st.info(ask_ai("Gemini", f"Сызықтық регрессияны градиенттік түсумен оқыттық: жылдамдық={lr}, қадам={steps_n}, "
                           f"шу={noise}, нәтиже w={w:.2f}, b={b:.2f}. 8-сынып оқушысына қарапайым түсіндір."))

# ---------- Курстар ----------
elif page == "📚 Курстар":
    st.header("📚 Курстар")
    course = st.selectbox("Курс", list(COURSES))
    lessons = COURSES[course]
    i = st.selectbox("Сабақ", range(len(lessons)), format_func=lambda k: lessons[k]["t"])
    L, key = lessons[i], f"{course}:{i}"
    st.subheader(L["t"])
    st.write(L["x"])
    st.code(L["c"], language="python")
    q, opts, right = L["q"]
    pick = st.radio(q, opts, index=None, key="q" + key)
    if st.button("Жауапты тексеру") and pick:
        if opts.index(pick) == right:
            if key not in ss.done:
                ss.xp += 20
            ss.done.add(key)
            st.success("Дұрыс! Сабақ аяқталды ✅")
        else:
            st.error("Қате. Мәтінді қайта оқып көріңіз.")
    if key in ss.done:
        st.caption("✅ Бұл сабақ аяқталған")
    prov = st.selectbox("AI-дан сұрау", list(ENV), key="cp")
    qq = st.text_input("Сабақ бойынша сұрағыңыз")
    if st.button("Сұрау") and qq:
        st.info(ask_ai(prov, f"Сабақ: {L['t']}. {L['x']}\nСұрақ: {qq}"))

# ---------- AI чат ----------
else:
    st.header("🤖 AI чат")
    choice = st.selectbox("Қай AI?", list(ENV))
    for who, text in ss.chat:
        st.chat_message("user" if who == "user" else "assistant").write(text)
    if msg := st.chat_input("Сұрағыңызды жазыңыз"):
        ss.chat.append(("user", msg))
        st.chat_message("user").write(msg)
        if choice == "Үшеуін салыстыру":
            cols = st.columns(3)
            out = []
            for col, n in zip(cols, ENV):
                with col:
                    st.subheader(n)
                    r = ask_ai(n, msg)
                    st.write(r)
                    out.append(f"**{n}:** {r}")
            ss.chat.append(("ai", "\n\n".join(out)))
        else:
            r = ask_ai(choice, msg)
            st.chat_message("assistant").write(r)
            ss.chat.append(("ai", r))
    if st.button("Чатты тазалау"):
        ss.chat = []
        st.rerun()
