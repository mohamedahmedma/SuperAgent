"""The second regression suite: a parent's own corpus, single turns and conversations.

`planner_execution_eval.py` is the first suite and it was written alongside the tools, so
it inherits their vocabulary — it tests the system in the words the system was built with.
This file is the opposite and that is its whole value: the questions were written down
independently, by somebody describing what parents actually send, and then handed over. A
suite authored by the same hand that wrote the prompt catalogue will always flatter it.

Everything about the harness is reused — `run_case`, the stubbed facade, the canned
retrieval, the stub answer model, the reporting. Only three things differ:

**A roster where the name in the corpus is unambiguous.** The corpus talks about "أحمد",
and the first suite's roster has `أحمد` as the shared SURNAME of both children — so every
case naming him would resolve to two children and the turn would correctly stop to ask
which. That is a property of that fixture, not of the question, so this suite seeds a boy
called أحمد محمود and a sister called سارة محمود. Both genders are on the roll because the
corpus says "ابني" and "بنتي" and "my daughter", and each of those has to land somewhere.

**Conversations are first-class.** Sections 17 and 18 of the corpus are multi-turn, and a
follow-up is where this system is weakest and most interesting: the subject lives in the
previous turn, not in the message. Each conversation contributes one case per follow-up,
with the turns before it as history.

**Expectations follow the corpus, not my preference.** Where the author stated the answer
("→ SIS grades", "→ RAG"), that is pinned even where I would have hedged: the point of a
handed-over suite is to find out whether the system agrees with somebody who was not
looking at the code. Where the author themselves marked a case ambiguous or "potentially",
it is `observe_only` — pinning a coin-flip as a requirement is how a suite starts lying.

Run it the same way, and it takes the same flags:

    ACTIVE_PROFILE=school python -m tests.evals.parent_regression_eval --parallel=16
    ACTIVE_PROFILE=school python -m tests.evals.parent_regression_eval --only="conv"
"""
from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import tests.evals.planner_execution_eval as pe
from tests.evals.planner_execution_eval import (
    A,
    C,
    Case,
    G,
    K,
    S,
    SUBJ,
    T,
    TCH,
    TSUB,
    _fake_rag,
    _load_report,
    _RATE_LIMITS,
    _records_get,
    _report,
    _roster_fetch,
    run_case,
)
from backend.profiles.registry import get_profile

# ---------------------------------------------------------------------------
# The family this suite talks about.
# ---------------------------------------------------------------------------
#
# Assigned onto the harness module rather than passed in, because `_roster_fetch` and the
# stranger-check inside `run_case` both read that module's global at call time. One
# assignment here re-points both, and the first suite is unaffected because the two never
# run in the same process.
ROSTER = [
    {
        "student_id": "S-100",
        "full_name_ar": "أحمد محمود",
        "full_name_en": "Ahmed Mahmoud",
        "gender": "male",
        "year_level": "Primary 4",
    },
    {
        "student_id": "S-200",
        "full_name_ar": "سارة محمود",
        "full_name_en": "Sara Mahmoud",
        "gender": "female",
        "year_level": "Primary 1",
    },
]
pe.ROSTER = ROSTER


#: Short, plausible assistant turns for the conversation cases. The resolver and the
#: classifier both READ this text, so a placeholder like "[grades]" would be answering a
#: different question from the one a parent actually saw.
_SAID_GRADES = "درجات أحمد: الرياضيات ٨٧٪، العلوم ٩١٪."
_SAID_TEACHERS = "مدرسين فصل أحمد: أ. سامي للرياضيات، وأ. هدى للعلوم."
_SAID_TIMETABLE = "جدول أحمد: الأحد حصة رياضيات، الاثنين علوم."
_SAID_ATTENDANCE = "أحمد حضر ٥٨ يوم وغاب يومين."
_SAID_FEES = "مصاريف السنة ٤٠ ألف جنيه."
_SAID_SUBJECTS = "أحمد بيدرس: الرياضيات، العلوم، اللغة العربية."
_SAID_CLASS = "أحمد في فصل الرابع/١."
_SAID_BUS = "أيوة، المدرسة عندها باصات."
_SAID_PAYMENT = "الدفع بالتحويل البنكي أو في مكتب المدرسة."
_SAID_GRADE5 = "أيوة، المدرسة بتقبل في Grade 5."


CASES = [
    # =====================================================================================
    # 1. SIS — درجات الطالب
    # =====================================================================================
    Case("grades — how much", "كم درجات أحمد؟", expect_ran={G}, technique="grades"),
    Case("grades — my son", "عايزة أعرف درجات ابني.", expect_ran={G}, technique="grades"),
    Case("grades — English, daughter", "What are my daughter's grades?", expect_ran={G},
         technique="grades"),
    Case("grades — English, exams", "How did Ahmed do in his exams?",
         expect_ran_includes={G}, technique="grades",
         note="'exams' could pull toward the published exam schedule; the marks half is "
              "what is asserted"),
    Case("grades — result of the student", "ممكن أعرف نتيجة الطالب أحمد؟", expect_ran={G},
         technique="grades"),
    Case("grades — his marks per subject", "درجاته في المواد كام؟", observe_only=True,
         technique="grades",
         note="carried entirely by the enclitic 'ـه' with no conversation behind it. "
              "Asking which child is the correct outcome; answering about one is not"),
    Case("grades — all of them", "عايز أشوف درجات ابني كلها.", expect_ran={G},
         technique="grades"),
    Case("grades — English, performing", "How is my son performing academically?",
         expect_ran_includes={G}, technique="grades"),
    Case("grades — one subject, maths", "ابني جاب كام في الرياضيات؟", expect_ran={S},
         model_args={S: {"subject": "الرياضيات"}}, technique="grades"),
    Case("grades — one subject, English", "كم حصلت بنتي في English؟", expect_ran={S},
         model_args={S: {"subject": "English"}}, technique="grades"),
    Case("grades — this term", "ممكن توريني درجاته في الترم ده؟", observe_only=True,
         technique="grades", note="enclitic pronoun, no history — see 'his marks per subject'"),
    Case("grades — English, this term", "What marks does my child have this term?",
         observe_only=True, technique="grades",
         note="'my child' picks neither of the two on the roll — unlike 'my son' or 'my "
              "daughter', which the gender resolves. Asking which is correct here"),
    Case("grades — level", "عايز أعرف مستوى درجات الطالب.", observe_only=True,
         technique="grades",
         note="'الطالب' names no child of this parent's; asking which is right"),
    Case("grades — last exam", "درجات أحمد في آخر امتحان إيه؟", expect_ran_includes={G},
         technique="grades",
         note="the contract has no per-exam breakdown, so the term marks are the honest "
              "answer; what matters is that it is read rather than invented"),
    Case("grades — result per subject", "هل ممكن أعرف نتيجة ابني في المواد؟",
         expect_ran={G}, technique="grades"),

    # Production-style, as the corpus calls them: shorter, more elliptical.
    Case("grades — dialect, math", "هو أحمد جاب كام في math؟", expect_ran={S},
         model_args={S: {"subject": "math"}}, technique="grades"),
    Case("grades — the rest of the subjects", "طيب ودرجات باقي المواد؟",
         history=("هو أحمد جاب كام في math؟", "أحمد جاب ٨٧٪ في الرياضيات."),
         expect_ran={G}, technique="grades",
         note="a follow-up widening from one subject to all of them"),
    Case("grades — and English", "و English عامل إيه؟",
         history=("هو أحمد جاب كام في math؟", "أحمد جاب ٨٧٪ في الرياضيات."),
         expect_ran={S}, model_args={S: {"subject": "English"}}, technique="grades"),
    Case("grades — is his level good", "طب هو مستواه كويس ولا لأ؟",
         history=("درجات أحمد كام؟", _SAID_GRADES),
         expect_ran_includes={G}, technique="grades"),
    Case("grades — in detail", "ممكن تقولي درجاته كلها بالتفصيل؟",
         history=("درجات أحمد كام؟", _SAID_GRADES),
         expect_ran_includes={G}, technique="grades"),
    Case("grades — has the result come out", "ابني نتيجته ظهرت؟", expect_ran={G},
         technique="grades"),
    Case("grades — any marks this term", "فيه درجات للترم الحالي؟", observe_only=True,
         technique="grades", note="names no child and none is pinned"),
    Case("grades — current", "عايز أعرف درجاته الحالية.", observe_only=True,
         technique="grades"),
    Case("grades — the marks, code-switched", "ممكن أشوف الـ marks بتاعته؟",
         observe_only=True, technique="grades"),
    Case("grades — English, so far", "What are his marks so far?", observe_only=True,
         technique="grades"),

    # =====================================================================================
    # 2. SIS — الحضور
    # =====================================================================================
    Case("attendance — days attended", "أحمد حضر كام يوم؟", expect_ran={A},
         technique="attendance"),
    Case("attendance — my son", "عايزة أعرف حضور ابني.", expect_ran={A},
         technique="attendance"),
    Case("attendance — English", "How many days has my daughter attended?", expect_ran={A},
         technique="attendance"),
    Case("attendance — recent absence", "هل أحمد غاب الأيام اللي فاتت؟", expect_ran={A},
         technique="attendance"),
    Case("attendance — how many absences", "كام يوم غياب عنده؟", observe_only=True,
         technique="attendance", note="enclitic only, no history"),
    Case("attendance — English, record", "What is my son's attendance record?",
         expect_ran={A}, technique="attendance"),
    Case("attendance — percentage", "ممكن أعرف نسبة الحضور؟", observe_only=True,
         technique="attendance", note="names no child"),
    Case("attendance — any absence", "هل عليه أي غياب؟", observe_only=True,
         technique="attendance"),
    Case("attendance — how many times", "ابني غاب كام مرة؟", expect_ran={A},
         technique="attendance"),
    Case("attendance — English, named", "How many absences does Ahmed have?",
         expect_ran={A}, technique="attendance"),
    Case("attendance — code-switched", "عايز أعرف attendance بتاع الطالب.",
         observe_only=True, technique="attendance"),
    Case("attendance — the register", "ممكن أشوف سجل الحضور؟", observe_only=True,
         technique="attendance"),
    Case("attendance — present today", "هو كان حاضر النهارده؟", observe_only=True,
         technique="attendance",
         note="the contract reports a term total, not one day. Reading it is right; "
              "answering 'yes, today' from it is not, and no expectation should imply the "
              "system holds a per-day answer for a bare pronoun"),
    Case("attendance — this month", "هل عنده غياب الشهر ده؟", observe_only=True,
         technique="attendance"),
    Case("attendance — English, percentage", "What is her attendance percentage?",
         observe_only=True, technique="attendance"),

    # =====================================================================================
    # 3. SIS — المواد
    # =====================================================================================
    Case("subjects — what does he study", "أحمد بيدرس مواد إيه؟", expect_ran={SUBJ},
         technique="subjects"),
    Case("subjects — MSA", "ما هي المواد التي يدرسها ابني؟", expect_ran={SUBJ},
         technique="subjects"),
    Case("subjects — English", "What subjects does my daughter study?", expect_ran={SUBJ},
         technique="subjects"),
    Case("subjects — code-switched", "ممكن أعرف الـ subjects بتاعت الطالب؟",
         observe_only=True, technique="subjects", note="names no child"),
    Case("subjects — how many", "ابني عنده كام مادة؟", expect_ran={SUBJ},
         technique="subjects"),
    Case("subjects — English, enrolled", "What subjects is Ahmed enrolled in?",
         expect_ran={SUBJ}, technique="subjects"),
    Case("subjects — the study subjects", "المواد الدراسية للطالب إيه؟", observe_only=True,
         technique="subjects"),
    Case("subjects — list", "عايز list المواد بتاعته.", observe_only=True,
         technique="subjects"),
    Case("subjects — English, classes taking", "What classes is my child taking?",
         observe_only=True, technique="subjects",
         note="two ambiguities at once: 'classes' is the subject list or the room, and 'my "
              "child' names neither of the two on the roll. Asking is the right outcome"),
    Case("subjects — this year", "إيه المواد اللي بياخدها أحمد السنة دي؟",
         expect_ran={SUBJ}, technique="subjects"),
    Case("subjects — natural, what does he study", "هو بيدرس إيه السنة دي؟",
         observe_only=True, technique="subjects"),
    Case("subjects — does he have science", "طب عنده science؟",
         history=("أحمد بيدرس مواد إيه؟", _SAID_SUBJECTS),
         expect_ran_includes={SUBJ}, technique="subjects"),
    Case("subjects — and French", "وهل بيدرس French؟",
         history=("أحمد بيدرس مواد إيه؟", _SAID_SUBJECTS),
         expect_ran_includes={SUBJ}, technique="subjects"),
    Case("subjects — all of them", "إيه كل المواد اللي عنده؟",
         history=("أحمد في فصل إيه؟", _SAID_CLASS),
         expect_ran={SUBJ}, technique="subjects"),
    Case("subjects — tell me his subjects", "ممكن تقولي المواد بتاعته؟",
         history=("أحمد في فصل إيه؟", _SAID_CLASS),
         expect_ran={SUBJ}, technique="subjects"),

    # =====================================================================================
    # 4. SIS — فصل الطالب
    # =====================================================================================
    Case("class — which class", "أحمد في فصل إيه؟", expect_ran={C}, technique="class"),
    Case("class — code-switched", "ابني في أنهي class؟", expect_ran={C}, technique="class"),
    Case("class — English", "What class is my daughter in?", expect_ran={C},
         technique="class"),
    Case("class — the student's class", "ممكن أعرف الفصل بتاع الطالب؟", observe_only=True,
         technique="class", note="names no child"),
    Case("class — English, assigned", "Which class is Ahmed assigned to?", expect_ran={C},
         technique="class"),
    Case("class — which primary", "هو في Primary كام؟", observe_only=True,
         technique="class",
         note="asks for the year group rather than the room; the class read carries both, "
              "but a bare pronoun means the child is not settled either"),
    Case("class — English, current", "What is my son's current class?", expect_ran={C},
         technique="class"),
    Case("class — the name", "عايز أعرف اسم الفصل.", observe_only=True, technique="class"),
    Case("class — belongs to which", "أحمد تبع أنهي فصل؟", expect_ran={C},
         technique="class"),
    Case("class — code-switched, his", "ممكن تقولي الـ class بتاعته؟", observe_only=True,
         technique="class"),

    # =====================================================================================
    # 5. SIS — مدرسين الفصل
    # =====================================================================================
    Case("teachers — of his class", "مين مدرسين الفصل بتاع أحمد؟", expect_ran={TCH},
         technique="teachers"),
    Case("teachers — English", "Who teaches my daughter's class?", expect_ran={TCH},
         technique="teachers"),
    Case("teachers — my son's class", "عايزة أعرف المدرسين بتوع فصل ابني.",
         expect_ran={TCH}, technique="teachers"),
    Case("teachers — who teaches the class", "مين المدرسين اللي بيدرسوا للفصل؟",
         observe_only=True, technique="teachers", note="names no child"),
    Case("teachers — English, assigned", "What teachers are assigned to my son's class?",
         expect_ran={TCH}, technique="teachers"),
    Case("teachers — names", "ممكن أعرف أسماء مدرسين الفصل؟", observe_only=True,
         technique="teachers"),
    Case("teachers — English, named child", "Who are Ahmed's teachers?", expect_ran={TCH},
         technique="teachers"),
    Case("teachers — list", "عايز قائمة المدرسين بتوع الفصل.", observe_only=True,
         technique="teachers"),
    Case("teachers — who is with the class", "مين المدرسين الموجودين مع الفصل؟",
         observe_only=True, technique="teachers"),
    Case("teachers — names of his class's", "إيه أسماء المدرسين اللي بيدرسوا للفصل بتاعه؟",
         observe_only=True, technique="teachers"),

    # =====================================================================================
    # 6. SIS — مدرس مادة معينة  (the corpus flags this as the one that must not collapse
    #    into section 5)
    # =====================================================================================
    Case("subject teacher — English, math", "Who teaches Math to my daughter?",
         expect_ran={TSUB}, model_args={TSUB: {"subject": "Math"}},
         technique="subject-teacher"),
    Case("subject teacher — English subject", "مين مدرس English لفصل ابني؟",
         expect_ran={TSUB}, model_args={TSUB: {"subject": "English"}},
         technique="subject-teacher"),
    Case("subject teacher — English, science", "Who is my son's science teacher?",
         expect_ran={TSUB}, model_args={TSUB: {"subject": "science"}},
         technique="subject-teacher"),
    Case("subject teacher — who teaches math to his class",
         "مين المدرس اللي بيدرس Math لفصل أحمد؟", expect_ran={TSUB},
         model_args={TSUB: {"subject": "Math"}}, technique="subject-teacher"),
    Case("subject teacher — Arabic", "مين مدرس العربي بتاعه؟", observe_only=True,
         technique="subject-teacher", note="enclitic only; the subject is clear, the child is not"),
    Case("subject teacher — English, Arabic subject", "Who teaches Arabic to Ahmed?",
         expect_ran={TSUB}, model_args={TSUB: {"subject": "Arabic"}},
         technique="subject-teacher"),
    Case("subject teacher — the science one", "ممكن أعرف مدرس الـ Science؟",
         observe_only=True, technique="subject-teacher", note="names no child"),
    Case("subject teacher — the maths subject", "مين مدرس مادة الرياضيات عند أحمد؟",
         expect_ran={TSUB}, model_args={TSUB: {"subject": "الرياضيات"}},
         technique="subject-teacher"),
    Case("subject teacher — English in my son's class", "مين مدرس English في فصل ابني؟",
         expect_ran={TSUB}, model_args={TSUB: {"subject": "English"}},
         technique="subject-teacher"),

    # The corpus's own boundary set: the four messages that decide whether the all-vs-one
    # split holds. This is the most load-bearing group in the file.
    Case("boundary — all his teachers", "مين مدرسين أحمد؟", expect_ran={TCH},
         technique="boundary"),
    Case("boundary — one subject's teacher", "مين مدرس الرياضيات بتاع أحمد؟",
         expect_ran={TSUB}, model_args={TSUB: {"subject": "الرياضيات"}},
         technique="boundary"),
    Case("boundary — singular, no subject named", "مين المدرس اللي بيدرس أحمد؟",
         observe_only=True, technique="boundary",
         note="the corpus marks this ambiguous itself: singular 'المدرس' with no subject. "
              "All his teachers is the defensible read; naming one at random is not"),
    Case("boundary — the school's maths teacher", "مين مدرس الرياضيات في المدرسة؟",
         observe_only=True, technique="boundary",
         note="the corpus says this may not be the student-specific query. A school-wide "
              "staff question belongs to the corpus; his class's belongs to the record"),
    Case("boundary — his class's, not the school's",
         "مين مدرس Math بتاع فصل أحمد مش كل مدرسين المدرسة؟", expect_ran={TSUB},
         model_args={TSUB: {"subject": "Math"}}, technique="boundary",
         note="states the distinction outright, so it is the one case in this group with "
              "no excuse for getting it wrong"),

    # =====================================================================================
    # 7. SIS — جدول الحصص
    # =====================================================================================
    Case("timetable — may I see", "ممكن أشوف جدول ابني؟", expect_ran={T},
         technique="timetable"),
    Case("timetable — English", "What is my daughter's timetable?", expect_ran={T},
         technique="timetable"),
    Case("timetable — English, tomorrow", "What classes does Ahmed have tomorrow?",
         expect_ran={T}, technique="timetable"),
    Case("timetable — the student's", "جدول الحصص بتاع الطالب؟", observe_only=True,
         technique="timetable", note="names no child"),
    Case("timetable — Sunday", "عنده حصص إيه يوم الأحد؟", observe_only=True,
         technique="timetable"),
    Case("timetable — English, Monday", "What lessons does my son have on Monday?",
         expect_ran={T}, technique="timetable"),
    Case("timetable — first lesson", "أول حصة عند أحمد إيه؟", expect_ran={T},
         technique="timetable"),
    Case("timetable — when is maths", "إمتى عنده Math؟", observe_only=True,
         technique="timetable"),
    Case("timetable — English, what time", "What time is English tomorrow?",
         observe_only=True, technique="timetable", note="names no child"),
    Case("timetable — how many lessons tomorrow", "عنده كام حصة بكرة؟", observe_only=True,
         technique="timetable"),
    Case("timetable — code-switched", "ممكن توريني timetable بتاعه؟", observe_only=True,
         technique="timetable"),
    Case("timetable — the week", "إيه جدول الأسبوع؟", observe_only=True,
         technique="timetable"),
    Case("timetable — science on Tuesday", "هل عنده Science يوم الثلاثاء؟",
         observe_only=True, technique="timetable"),
    Case("timetable — English, this week", "What is Ahmed's schedule for this week?",
         expect_ran={T}, technique="timetable"),

    # =====================================================================================
    # 8-14. RAG — the school's own published material
    # =====================================================================================
    Case("rag fees — how much", "كام مصاريف المدرسة؟", expect_ran={K},
         model_args={K: {"query": "مصاريف المدرسة"}}, technique="rag-fees"),
    Case("rag fees — English", "What are the school fees?", expect_ran={K},
         model_args={K: {"query": "school fees"}}, technique="rag-fees"),
    Case("rag fees — per year", "بكام السنة الدراسية؟", expect_ran={K},
         model_args={K: {"query": "مصاريف السنة الدراسية"}}, technique="rag-fees"),
    Case("rag fees — one grade", "مصاريف Primary 1 كام؟", expect_ran={K},
         model_args={K: {"query": "مصاريف Primary 1"}}, technique="rag-fees"),
    Case("rag fees — English, tuition", "How much does tuition cost?", expect_ran={K},
         model_args={K: {"query": "tuition cost"}}, technique="rag-fees"),
    Case("rag fees — the amount", "إيه قيمة المصروفات؟", expect_ran={K},
         model_args={K: {"query": "قيمة المصروفات"}}, technique="rag-fees"),
    Case("rag fees — vary by grade", "هل المصاريف بتختلف حسب الصف؟", expect_ran={K},
         model_args={K: {"query": "المصاريف حسب الصف"}}, technique="rag-fees"),
    Case("rag fees — English, this year", "What are the fees for this academic year?",
         expect_ran={K}, model_args={K: {"query": "fees this academic year"}},
         technique="rag-fees"),
    Case("rag fees — instalments", "هل فيه installment plan للمصاريف؟", expect_ran={K},
         model_args={K: {"query": "تقسيط المصاريف"}}, technique="rag-fees"),
    Case("rag fees — details", "ممكن أعرف تفاصيل المصروفات؟", expect_ran={K},
         model_args={K: {"query": "تفاصيل المصروفات"}}, technique="rag-fees"),

    Case("rag uniform — price", "الزي المدرسي بكام؟", expect_ran={K},
         model_args={K: {"query": "سعر الزي المدرسي"}}, technique="rag-uniform"),
    Case("rag uniform — where", "منين أجيب school uniform؟", expect_ran={K},
         model_args={K: {"query": "شراء الزي المدرسي"}}, technique="rag-uniform"),
    Case("rag uniform — English, policy", "What is the school uniform policy?",
         expect_ran={K}, model_args={K: {"query": "uniform policy"}},
         technique="rag-uniform"),
    Case("rag uniform — does the school sell it", "هل المدرسة بتبيع الزي؟", expect_ran={K},
         model_args={K: {"query": "بيع الزي المدرسي"}}, technique="rag-uniform"),
    Case("rag uniform — what is required", "إيه المطلوب في الـ uniform؟", expect_ran={K},
         model_args={K: {"query": "متطلبات الزي المدرسي"}}, technique="rag-uniform"),
    Case("rag uniform — compulsory", "هل فيه uniform إجباري؟", expect_ran={K},
         model_args={K: {"query": "الزي المدرسي إجباري"}}, technique="rag-uniform"),
    Case("rag uniform — English, where to get", "Where can I get the school uniform?",
         expect_ran={K}, model_args={K: {"query": "where to buy uniform"}},
         technique="rag-uniform"),
    Case("rag uniform — prices", "أسعار الزي المدرسي كام؟", expect_ran={K},
         model_args={K: {"query": "أسعار الزي المدرسي"}}, technique="rag-uniform"),

    Case("rag payment — how to pay", "إزاي أدفع المصاريف؟", expect_ran={K},
         model_args={K: {"query": "طرق دفع المصاريف"}}, technique="rag-payment"),
    Case("rag payment — English, methods", "What payment methods do you accept?",
         expect_ran={K}, model_args={K: {"query": "payment methods"}},
         technique="rag-payment"),
    Case("rag payment — online", "ينفع أدفع أونلاين؟", expect_ran={K},
         model_args={K: {"query": "الدفع أونلاين"}}, technique="rag-payment"),
    Case("rag payment — instalments", "هل فيه تقسيط؟", expect_ran={K},
         model_args={K: {"query": "التقسيط"}}, technique="rag-payment"),
    Case("rag payment — English, transfer", "Can I pay by bank transfer?", expect_ran={K},
         model_args={K: {"query": "bank transfer"}}, technique="rag-payment"),
    Case("rag payment — the methods", "إيه طرق دفع المصاريف؟", expect_ran={K},
         model_args={K: {"query": "طرق الدفع"}}, technique="rag-payment"),
    Case("rag payment — credit card", "هل بتقبلوا credit card؟", expect_ran={K},
         model_args={K: {"query": "الدفع بالكارت"}}, technique="rag-payment"),
    Case("rag payment — InstaPay", "أقدر أدفع عن طريق InstaPay؟", expect_ran={K},
         model_args={K: {"query": "InstaPay"}}, technique="rag-payment"),
    Case("rag payment — cash", "هل فيه دفع كاش؟", expect_ran={K},
         model_args={K: {"query": "الدفع كاش"}}, technique="rag-payment"),
    Case("rag payment — English, tuition", "How can I pay the tuition fees?",
         expect_ran={K}, model_args={K: {"query": "pay tuition"}}, technique="rag-payment"),

    Case("rag grades offered — from what age", "المدرسة بتقبل من سن كام؟", expect_ran={K},
         model_args={K: {"query": "سن القبول"}}, technique="rag-grades"),
    Case("rag grades offered — English", "What grades are available?", expect_ran={K},
         model_args={K: {"query": "grades available"}}, technique="rag-grades"),
    Case("rag grades offered — which grades", "إيه الصفوف اللي المدرسة بتقبلها؟",
         expect_ran={K}, model_args={K: {"query": "الصفوف المتاحة"}},
         technique="rag-grades"),
    Case("rag grades offered — KG", "هل فيه KG؟", expect_ran={K},
         model_args={K: {"query": "رياض الأطفال"}}, technique="rag-grades"),
    Case("rag grades offered — English, Primary 1", "Do you have Primary 1?",
         expect_ran={K}, model_args={K: {"query": "Primary 1"}}, technique="rag-grades"),
    Case("rag grades offered — secondary", "المدرسة فيها Secondary؟", expect_ran={K},
         model_args={K: {"query": "المرحلة الثانوية"}}, technique="rag-grades"),
    Case("rag grades offered — school years", "إيه السنوات الدراسية الموجودة؟",
         expect_ran={K}, model_args={K: {"query": "السنوات الدراسية"}},
         technique="rag-grades"),
    Case("rag grades offered — English, age groups", "What age groups do you accept?",
         expect_ran={K}, model_args={K: {"query": "age groups accepted"}},
         technique="rag-grades"),
    Case("rag grades offered — new students grade 6",
         "هل بتقبلوا طلاب جدد في Grade 6؟", expect_ran={K},
         model_args={K: {"query": "قبول طلاب جدد Grade 6"}}, technique="rag-grades"),

    Case("rag transport — is there a bus", "هل المدرسة عندها باص؟", expect_ran={K},
         model_args={K: {"query": "باص المدرسة"}}, technique="rag-transport"),
    Case("rag transport — English, options", "What transportation options are available?",
         expect_ran={K}, model_args={K: {"query": "transportation options"}},
         technique="rag-transport"),
    Case("rag transport — school bus", "هل فيه school bus؟", expect_ran={K},
         model_args={K: {"query": "school bus"}}, technique="rag-transport"),
    Case("rag transport — which areas", "الباص بيغطي مناطق إيه؟", expect_ran={K},
         model_args={K: {"query": "مناطق الباص"}}, technique="rag-transport"),
    Case("rag transport — cost", "إيه تكلفة الباص؟", expect_ran={K},
         model_args={K: {"query": "تكلفة الباص"}}, technique="rag-transport"),
    Case("rag transport — English, how it works",
         "How does the school bus service work?", expect_ran={K},
         model_args={K: {"query": "school bus service"}}, technique="rag-transport"),
    Case("rag transport — is there transport", "هل فيه مواصلات للمدرسة؟", expect_ran={K},
         model_args={K: {"query": "مواصلات المدرسة"}}, technique="rag-transport"),
    Case("rag transport — all areas", "هل الباص متاح لكل المناطق؟", expect_ran={K},
         model_args={K: {"query": "تغطية الباص"}}, technique="rag-transport"),
    Case("rag transport — can I subscribe", "أقدر أشترك في الباص؟", expect_ran={K},
         model_args={K: {"query": "الاشتراك في الباص"}}, technique="rag-transport"),

    Case("rag contact — whatsapp", "رقم واتساب المدرسة إيه؟", expect_ran={K},
         model_args={K: {"query": "رقم واتساب المدرسة"}}, technique="rag-contact"),
    Case("rag contact — English", "How can I contact the school?", expect_ran={K},
         model_args={K: {"query": "contact the school"}}, technique="rag-contact"),
    Case("rag contact — the whatsapp number", "ممكن رقم الواتساب؟", expect_ran={K},
         model_args={K: {"query": "رقم الواتساب"}}, technique="rag-contact"),
    Case("rag contact — English, whatsapp", "What is the school's WhatsApp number?",
         expect_ran={K}, model_args={K: {"query": "WhatsApp number"}},
         technique="rag-contact"),
    Case("rag contact — ways to reach", "إيه طرق التواصل مع المدرسة؟", expect_ran={K},
         model_args={K: {"query": "طرق التواصل"}}, technique="rag-contact"),
    Case("rag contact — do you have whatsapp", "عندكم WhatsApp؟", expect_ran={K},
         model_args={K: {"query": "واتساب المدرسة"}}, technique="rag-contact"),
    Case("rag contact — email", "إيميل المدرسة إيه؟", expect_ran={K},
         model_args={K: {"query": "إيميل المدرسة"}}, technique="rag-contact"),
    Case("rag contact — English, admissions", "How can I reach admissions?",
         expect_ran={K}, model_args={K: {"query": "admissions contact"}},
         technique="rag-contact"),
    Case("rag contact — phone", "رقم التليفون بتاع المدرسة كام؟", expect_ran={K},
         model_args={K: {"query": "رقم تليفون المدرسة"}}, technique="rag-contact"),
    Case("rag contact — who to ask", "مين أتواصل معاه لو عندي استفسار؟", expect_ran={K},
         model_args={K: {"query": "التواصل للاستفسارات"}}, technique="rag-contact"),

    Case("rag general — opening time", "المدرسة بتفتح الساعة كام؟", expect_ran={K},
         model_args={K: {"query": "مواعيد المدرسة"}}, technique="rag-general"),
    Case("rag general — English, location", "Where is the school located?",
         expect_ran={K}, model_args={K: {"query": "school location"}},
         technique="rag-general"),
    Case("rag general — address", "إيه عنوان المدرسة؟", expect_ran={K},
         model_args={K: {"query": "عنوان المدرسة"}}, technique="rag-general"),
    Case("rag general — English, curriculum", "What curriculum does the school follow?",
         expect_ran={K}, model_args={K: {"query": "curriculum"}}, technique="rag-general"),
    Case("rag general — american curriculum", "هل المدرسة بتدرس American curriculum؟",
         expect_ran={K}, model_args={K: {"query": "المنهج الأمريكي"}},
         technique="rag-general"),
    Case("rag general — start of year", "إمتى يبدأ العام الدراسي؟", expect_ran={K},
         model_args={K: {"query": "بداية العام الدراسي"}}, technique="rag-general"),
    Case("rag general — English, languages", "What languages are taught?", expect_ran={K},
         model_args={K: {"query": "languages taught"}}, technique="rag-general"),
    Case("rag general — mixed school", "المدرسة مختلطة؟", expect_ran={K},
         model_args={K: {"query": "مدرسة مختلطة"}}, technique="rag-general"),
    Case("rag general — the system", "إيه نظام الدراسة في المدرسة؟", expect_ran={K},
         model_args={K: {"query": "نظام الدراسة"}}, technique="rag-general"),
    Case("rag general — general info", "ممكن تديني معلومات عامة عن المدرسة؟",
         expect_ran={K}, model_args={K: {"query": "معلومات عن المدرسة"}},
         technique="rag-general"),

    # =====================================================================================
    # 15. The corpus's own mandatory regression set — SIS vs RAG
    # =====================================================================================
    #
    # Each pair is two messages that differ by one word and must go to different systems.
    # This is the set the author called mandatory, so the stated answer is pinned even
    # where I would have hedged — the value of a handed-over suite is finding out whether
    # the system agrees with somebody who was not reading the code.
    Case("vs — fees for the child, not marks", "كام مصاريف أحمد؟", expect_ran={K},
         model_args={K: {"query": "المصاريف"}}, technique="sis-vs-rag",
         note="names a child and asks about money. The corpus says fees; a records read "
              "here is an audited read of a minor's file that answers nothing"),
    Case("vs — marks for the child", "كام درجة أحمد؟", expect_ran={G},
         technique="sis-vs-rag"),
    Case("vs — what the child studies", "أحمد بيدرس إيه؟", expect_ran={SUBJ},
         technique="sis-vs-rag"),
    Case("vs — what the school teaches", "المدرسة بتدرس إيه؟", expect_ran={K},
         model_args={K: {"query": "المناهج"}}, technique="sis-vs-rag",
         note="the same verb, the subject swapped from the child to the school"),
    Case("vs — the child's teacher", "مين مدرس أحمد؟", expect_ran_includes={TCH},
         technique="sis-vs-rag",
         note="the corpus says SIS. Singular phrasing with no subject named means the "
              "class's staff; only the family is asserted, not which of the two tools"),
    Case("vs — the school's teachers", "مين مدرسين المدرسة؟", expect_ran={K},
         model_args={K: {"query": "مدرسين المدرسة"}}, technique="sis-vs-rag"),
    Case("vs — does the child have maths tomorrow", "هل أحمد عنده Math بكرة؟",
         expect_ran={T}, technique="sis-vs-rag"),
    Case("vs — how maths is taught at the school", "Math في المدرسة بتتدرس إزاي؟",
         observe_only=True, technique="sis-vs-rag",
         note="the corpus says 'potentially RAG' and hedges, so this is observed"),
    Case("vs — was the child present today", "هل أحمد حضر النهارده؟", expect_ran={A},
         technique="sis-vs-rag"),
    Case("vs — the school's hours", "مواعيد المدرسة إيه؟", expect_ran={K},
         model_args={K: {"query": "مواعيد المدرسة"}}, technique="sis-vs-rag"),
    Case("vs — the school's schedule", "جدول المدرسة إيه؟", observe_only=True,
         technique="sis-vs-rag",
         note="the corpus marks this one ambiguous itself — school-wide timings versus a "
              "child's week"),
    Case("vs — the child's schedule", "جدول أحمد إيه؟", expect_ran={T},
         technique="sis-vs-rag"),

    # =====================================================================================
    # 16. Out of domain — neither system may be reached
    # =====================================================================================
    Case("ood — weather", "الجو عامل إيه النهارده؟", expect_short_circuit=True,
         technique="out-of-domain"),
    Case("ood — football", "مين كسب ماتش الأهلي؟", expect_short_circuit=True,
         technique="out-of-domain"),
    Case("ood — write a poem", "اكتبلي شعر عن المدرسة.", expect_short_circuit=True,
         technique="out-of-domain",
         note="mentions the school, which is the trap: the topic is not the errand"),
    Case("ood — capital of France", "What is the capital of France?",
         expect_short_circuit=True, technique="out-of-domain"),
    Case("ood — solve an equation", "حللي المعادلة دي.", expect_short_circuit=True,
         technique="out-of-domain",
         note="homework help. Adjacent to a school and still not this assistant's job"),
    Case("ood — write code", "اكتبلي Python code.", expect_short_circuit=True,
         technique="out-of-domain"),
    Case("ood — best footballer", "مين أحسن لاعب كرة في العالم؟",
         expect_short_circuit=True, technique="out-of-domain"),
    Case("ood — tell a joke", "احكيلي نكتة.", expect_short_circuit=True,
         technique="out-of-domain"),
    Case("ood — translate", "ترجم الجملة دي للإنجليزي.", expect_short_circuit=True,
         technique="out-of-domain"),
    Case("ood — what is AI", "What is artificial intelligence?",
         expect_short_circuit=True, technique="out-of-domain"),

    # =====================================================================================
    # 17. Conversations — the follow-up carries its subject in the turn before
    # =====================================================================================
    Case("conv1 — grades then one subject", "طب والرياضيات؟",
         history=("عايز درجات أحمد.", _SAID_GRADES),
         expect_ran={S}, model_args={S: {"subject": "الرياضيات"}}, technique="conversation"),
    Case("conv2 — all teachers then one subject's", "طب مين مدرس الـ Math؟",
         history=("مين مدرسين أحمد؟", _SAID_TEACHERS),
         expect_ran={TSUB}, model_args={TSUB: {"subject": "Math"}},
         technique="conversation",
         note="the all-to-one edge as a follow-up — the pair section 6 exists to protect"),
    Case("conv3 — timetable then one day", "يوم الأحد بس.",
         history=("عايز جدول أحمد.", _SAID_TIMETABLE),
         expect_ran_includes={T}, technique="conversation",
         note="a fragment that is a filter on the previous answer, not a new question"),
    Case("conv4 — attendance then one month", "الشهر ده بس.",
         history=("أحمد غاب كام يوم؟", _SAID_ATTENDANCE),
         expect_ran_includes={A}, technique="conversation"),
    Case("conv5 — fees then instalments", "طيب ينفع أقسط؟",
         history=("مصاريف المدرسة كام؟", _SAID_FEES),
         expect_ran={K}, model_args={K: {"query": "التقسيط"}}, technique="conversation"),
    Case("conv6 — subjects then who teaches one", "ومين بيدرسله Science؟",
         history=("إيه المواد بتاعة أحمد؟", _SAID_SUBJECTS),
         expect_ran={TSUB}, model_args={TSUB: {"subject": "Science"}},
         technique="conversation"),
    Case("conv7 — class then its teachers", "ومين المدرسين بتوعه؟",
         history=("أحمد في فصل إيه؟", _SAID_CLASS),
         expect_ran={TCH}, technique="conversation"),
    Case("conv8 — timetable then the teacher", "ومين المدرس؟",
         history=("عنده Math إمتى؟", _SAID_TIMETABLE),
         observe_only=True, technique="conversation",
         note="the subject is two turns back and never in this message. Reading the "
              "class's teachers or asking which subject are both defensible; naming a "
              "teacher for a subject nobody restated is not"),
    Case("conv9 — bus then a district", "طيب بيعدي من مدينة نصر؟",
         history=("المدرسة فيها باص؟", _SAID_BUS),
         expect_ran={K}, model_args={K: {"query": "الباص مدينة نصر"}},
         technique="conversation"),
    Case("conv10 — payment then a method", "طب ينفع Visa؟",
         history=("إيه طرق الدفع؟", _SAID_PAYMENT),
         expect_ran={K}, model_args={K: {"query": "الدفع بفيزا"}}, technique="conversation"),

    # =====================================================================================
    # 18. Harder conversations — ellipsis, and switching side mid-thread
    # =====================================================================================
    Case("conv11a — ellipsis, one word subject", "English؟",
         history=("ممكن أعرف درجات ابني؟", _SAID_GRADES),
         expect_ran={S}, model_args={S: {"subject": "English"}}, technique="conversation",
         note="one word carrying a whole question. Everything that makes it answerable is "
              "in the previous turn"),
    Case("conv11b — ellipsis, second hop", "والـ Math؟",
         history=(
             "ممكن أعرف درجات ابني؟", _SAID_GRADES,
             "English؟", "درجة أحمد في اللغة الإنجليزية ٨٤٪.",
         ),
         expect_ran={S}, model_args={S: {"subject": "Math"}}, technique="conversation",
         note="two hops deep, and the subject changes on each — the state has to survive "
              "more than one turn"),
    Case("conv12 — attendance then a subject teacher", "طيب مين مدرس الـ Math بتاعه؟",
         history=("أحمد غاب كام يوم؟", _SAID_ATTENDANCE),
         expect_ran={TSUB}, model_args={TSUB: {"subject": "Math"}},
         technique="conversation",
         note="a domain switch inside the record family: attendance to staffing"),
    Case("conv13 — RAG then SIS", "طيب أحمد في Grade كام؟",
         history=("المدرسة بتقبل Grade 5؟", _SAID_GRADE5),
         expect_ran={C}, technique="conversation",
         note="crosses from published material to the child's own record, with the word "
              "'Grade' meaning something different on each side"),
    Case("conv14 — SIS then RAG", "والمصاريف كام؟",
         history=("أحمد بيدرس كام مادة؟", _SAID_SUBJECTS),
         expect_ran={K}, model_args={K: {"query": "المصاريف"}}, technique="conversation",
         note="the other direction, and the one the corpus's own history says used to "
              "carry the previous subject into the search"),

    # Section 15 of the corpus, asked as four consecutive messages rather than four
    # standalone ones: the same words, the subject moving between the child and the school.
    Case("conv15a — the child's teacher", "المدرس بتاع أحمد مين؟",
         expect_ran_includes={TCH}, technique="conversation"),
    Case("conv15b — the school's teachers", "المدرسين في المدرسة مين؟",
         history=("المدرس بتاع أحمد مين؟", _SAID_TEACHERS),
         expect_ran={K}, model_args={K: {"query": "مدرسين المدرسة"}},
         technique="conversation",
         note="the hard one: the previous turn was about this child, and the pull is to "
              "stay there. The subject moved to the school"),
    Case("conv15c — the child's maths teacher", "مين مدرس الـ Math بتاع أحمد؟",
         history=(
             "المدرس بتاع أحمد مين؟", _SAID_TEACHERS,
             "المدرسين في المدرسة مين؟", "المدرسة عندها ٤٥ مدرس.",
         ),
         expect_ran={TSUB}, model_args={TSUB: {"subject": "Math"}},
         technique="conversation",
         note="back to the child after a school-wide turn, which is the return leg of the "
              "switch above"),
    Case("conv15d — the school's maths teacher", "مين مدرس الـ Math في المدرسة؟",
         history=("مين مدرس الـ Math بتاع أحمد؟", "مدرس الرياضيات لأحمد هو أ. سامي."),
         observe_only=True, technique="conversation",
         note="the corpus marks this ambiguous and so is it: one word — بتاع أحمد versus "
              "في المدرسة — separates a record from the corpus, immediately after the "
              "record answer"),
]


def main() -> int:
    """Same harness, same flags, a different corpus. See the module docstring."""
    from backend.chat import child_roster
    from backend.tools import records

    records._get = _records_get
    child_roster._fetch = _roster_fetch

    import backend.rag.pipeline as pipeline

    pipeline.run_rag_graph = _fake_rag

    profile = get_profile()
    print(f"profile      {profile.name}")
    print(f"suite        parent regression corpus (second suite)")
    print(f"roster       {[row['full_name_ar'] for row in ROSTER]}")
    print(f"model        FAST_MODEL={os.getenv('FAST_MODEL')} (classifier + resolver)")

    workers = 1
    wanted = ""
    for arg in sys.argv[1:]:
        if arg.startswith("--parallel"):
            _, _, value = arg.partition("=")
            workers = int(value) if value else len(CASES)
        elif arg.startswith("--only"):
            _, _, wanted = arg.partition("=")

    cases = CASES
    if wanted:
        cases = [case for case in CASES if wanted.lower() in case.name.lower()]
        if not cases:
            print(f"no case name contains {wanted!r}")
            return 1
        print(f"filter       --only={wanted} matched {len(cases)} of {len(CASES)}")
        if workers >= len(CASES):
            workers = len(cases)
    print(f"cases        {len(cases)}")
    print(f"concurrency  {workers} turn(s) in flight")

    logging.getLogger().addHandler(_RATE_LIMITS)

    started = time.monotonic()
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run_case, cases))
    else:
        results = [run_case(case) for case in cases]
    wall_ms = int((time.monotonic() - started) * 1000)

    failed = _report(results)
    _load_report(results, workers, wall_ms)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
