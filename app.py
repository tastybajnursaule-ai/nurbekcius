"""AI Math Assistant — Python, Streamlit, SymPy.
Іске қосу: streamlit run app.py
"""
import os
import random
import re

import streamlit as st
from sympy import (Eq, Poly, Rational, S, solve, sqrt, symbols, nsimplify,
                   sympify, expand)
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
                                        implicit_multiplication_application,
                                        convert_xor)

x = symbols("x")
TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
SUP = str.maketrans("²³", "23")


# ---------- 1. Енгізуді оқу ----------
def parse_equation(text: str):
    t = text.translate(SUP).replace("²", "^2").replace("−", "-").replace(",", ".")
    t = re.sub(r"(?<=[a-z])2\b", "^2", t) if "^" not in t and "**" not in t else t
    if "=" not in t:
        raise ValueError("Теңдеуде «=» белгісі болуы керек. Мысалы: 2x^2 - 5x + 2 = 0")
    left, right = t.split("=", 1)
    return parse_expr(left, transformations=TRANSFORMS) - parse_expr(right, transformations=TRANSFORMS)


# ---------- 2. Есептің түрін анықтау ----------
def classify(expr):
    p = Poly(expand(expr), x)
    return {1: "Сызықтық теңдеу", 2: "Квадрат теңдеу"}.get(p.degree(), f"{p.degree()}-дәрежелі көпмүшелік теңдеу"), p


# ---------- 3-6. Шешу және қадамдар ----------
def solve_with_steps(expr):
    kind, p = classify(expr)
    steps, roots = [], []
    deg = p.degree()
    if deg == 1:
        a, b = p.all_coeffs()
        steps += [f"Әдіс: белгісізді бір жаққа шығару. Теңдеу: {a}x + ({b}) = 0",
                  f"{a}x = {-b}", f"x = {-b}/{a} = {Rational(-b, a)}"]
        roots = [Rational(-b, a)]
    elif deg == 2:
        a, b, c = p.all_coeffs()
        D = b**2 - 4*a*c
        steps += ["Әдіс: дискриминант. Формула: D = b² − 4ac",
                  f"a = {a}, b = {b}, c = {c}",
                  f"D = ({b})² − 4·({a})·({c}) = {D}"]
        if D > 0:
            r1, r2 = (-b + sqrt(D)) / (2*a), (-b - sqrt(D)) / (2*a)
            steps += ["D > 0, сондықтан екі түбір бар: x = (−b ± √D) / (2a)",
                      f"x₁ = {r1.simplify()}", f"x₂ = {r2.simplify()}"]
            roots = [r1.simplify(), r2.simplify()]
        elif D == 0:
            r = Rational(-b, 2*a)
            steps += ["D = 0, сондықтан бір түбір бар: x = −b / (2a)", f"x = {r}"]
            roots = [r]
        else:
            steps += ["D < 0, нақты түбір жоқ."]
    else:
        roots = [r for r in solve(expr, x) if r.is_real]
        steps += ["Әдіс: көбейткіштерге жіктеу / сандық шешу (SymPy)", f"Нақты түбірлер: {roots}"]
    return kind, steps, roots


# ---------- 7-8. Жауапты тексеру және қатені түсіндіру ----------
def read_answer(text: str):
    t = re.sub(r"x\s*\d*\s*=", "", text.translate(SUP).replace(",", " "))
    vals = [v for v in re.split(r"[;\s]+", t.strip()) if v]
    return {nsimplify(sympify(v)) for v in vals}


def explain_mistake(expr, user):
    _, p = classify(expr)
    if p.degree() == 2:
        a, b, c = p.all_coeffs()
        D = b**2 - 4*a*c
        if D >= 0:
            sq = sqrt(D)
            patterns = {
                "Таңбаны қате алдыңыз: формулада −b тұр, ал сіз b алған сияқтысыз.":
                    {(b + sq) / (2*a), (b - sq) / (2*a)},
                "2a-ға бөлуді ұмытқан сияқтысыз: бөлгіш a емес, 2a болуы керек.":
                    {(-b + sq) / a, (-b - sq) / a},
                "Дискриминантты қате санаған болуыңыз мүмкін: D = b² − 4ac, ал 4ac-тың таңбасын тексеріңіз.":
                    {(-b + sqrt(abs(b**2 + 4*a*c))) / (2*a), (-b - sqrt(abs(b**2 + 4*a*c))) / (2*a)},
            }
            for msg, s in patterns.items():
                if {nsimplify(v) for v in s} == user:
                    return msg
    return "Алған түбірлеріңізді теңдеуге қойып көріңіз: екі жағы тең болмаса, сол түбір қате."


def check_answer(expr, user_text):
    _, _, roots = solve_with_steps(expr)
    truth = {nsimplify(r) for r in roots}
    try:
        user = read_answer(user_text)
    except Exception:
        return False, "Жауабыңызды оқи алмадым. Мысалы: 2; 0.5"
    if user == truth:
        return True, "Дұрыс! Барлық түбірді таптыңыз."
    if user < truth:
        return False, "Жауабыңыз толық емес: кейбір түбір жетіспейді."
    return False, explain_mistake(expr, user)


# ---------- 9. Ұқсас жаңа есеп ----------
def new_problem(kind):
    if kind.startswith("Сызықтық"):
        a, b, c = random.randint(2, 9), random.randint(-9, 9), random.randint(-20, 20)
        return f"{a}x + {b} = {c}".replace("+ -", "- ")
    r1, r2, a = random.randint(-6, 6), random.randint(-6, 6), random.choice([1, 1, 2, 3])
    p = Poly(expand(a * (x - r1) * (x - r2)), x).all_coeffs()
    return f"{p[0]}x^2 + ({p[1]})x + ({p[2]}) = 0"


# ---------- AI түсіндірмесі (міндетті емес) ----------
def ai_explain(problem, steps):
    key = os.getenv("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", None)
    if not key:
        return None
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    r = client.messages.create(
        model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"), max_tokens=600,
        messages=[{"role": "user", "content":
                   f"Қазақ тілінде 8-сынып оқушысына есептің шешуін қысқа, қарапайым түсіндір.\n"
                   f"Есеп: {problem}\nШешу қадамдары:\n" + "\n".join(steps)}])
    return r.content[0].text


# ---------- Қосымша: сурет арқылы енгізу ----------
def read_image(file):
    try:
        import cv2
        import numpy as np
        import pytesseract
    except ImportError:
        return None, "Сурет режимі үшін: pip install opencv-python pytesseract (және Tesseract бағдарламасы)"
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return pytesseract.image_to_string(img, config="--psm 7").strip(), None


# ---------- Интерфейс ----------
st.set_page_config(page_title="AI Math Assistant", page_icon="🧮")
st.title("🧮 AI Math Assistant")
st.caption("Теңдеуді енгізіңіз, ал көмекші түрін анықтап, шешу жолын қадамдап түсіндіреді.")

if "problem" not in st.session_state:
    st.session_state.problem = "2x^2 - 5x + 2 = 0"

up = st.file_uploader("Немесе есептің суретін жүктеңіз (міндетті емес)", type=["png", "jpg", "jpeg"])
if up:
    text, err = read_image(up)
    if err:
        st.warning(err)
    elif text:
        st.session_state.problem = text

problem = st.text_input("Есепті енгізіңіз", key="problem")

try:
    expr = parse_equation(problem)
    kind, steps, roots = solve_with_steps(expr)
except Exception as e:
    st.error(f"Есепті түсінбедім. {e}")
    st.stop()

st.subheader(f"Есеп түрі: {kind}")
for i, s in enumerate(steps, 1):
    st.write(f"**{i}.** {s}")
st.success("Жауабы: " + (", ".join(f"x{i+1} = {r}" for i, r in enumerate(roots)) if roots else "нақты түбір жоқ"))

if st.button("AI түсіндірмесін алу"):
    out = ai_explain(problem, steps)
    st.info(out or "AI түсіндірмесі үшін ANTHROPIC_API_KEY қосыңыз. Қадамдар SymPy арқылы есептелді.")

st.divider()
ans = st.text_input("Менің жауабымды тексер", placeholder="Мысалы: 2; 0.5")
if st.button("Тексеру") and ans:
    ok, msg = check_answer(expr, ans)
    (st.success if ok else st.error)(msg)

if st.button("Осыған ұқсас есеп бер"):
    st.session_state.problem = new_problem(kind)
    st.rerun()
