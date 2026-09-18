"""The labelled school question set: what a parent asks, and what must be retrieved to answer it.

The corpus is `Aurexis_Knowledge_Base_Mock_Egypt.docx` and it is written in ENGLISH. Every
question here is in ARABIC, because that is what parents send. So each case measures the
cross-language path end to end — Arabic question, English evidence — which is the case a
same-language eval cannot see at all.

## What a case asserts

`required` is the evidence that MUST be retrieved, not a hint. A string must appear
somewhere in the retrieved set; a tuple means any one of its members will do, for facts the
corpus states in more than one place. The score that matters is therefore "did ALL of the
required evidence arrive", not "did something matching arrive" — a question answered by a
fee number AND its year-group row is only answerable when both are present, and a hit-rate
oracle that stops at the first match calls that a pass.

Every span was read out of the source document, so a miss is a retrieval failure rather
than a mislabelled oracle. Spans are deliberately short and free of the characters Word
renders inconsistently (en dashes, non-breaking spaces): `"105,000 EGP"`, not the whole
table row.

## Figures

`modality="figure"` marks a question whose answer is only inside an image — the uniform
photographs. Those cases carry no `required` spans on purpose: the text is whatever the
vision model transcribed at ingest, which is not knowable from the source file and changes
when the extraction prompt or model changes. They pass when a figure chunk is retrieved,
and the harness reports separately whether that figure's text survives into the grader's
view, which is the measurement the whole image-chunking fix is judged by.

## Splits

`holdout=True` marks roughly a fifth of the set. Tune against the rest; touch the holdout
only to confirm a finished change, or it stops being a holdout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

#: Bump when a case is added, removed, or its gold changes — a score is only comparable
#: against another run of the same version.
DATASET_VERSION = "2026-09-18.1"

#: The corpus these questions are labelled against.
CORPUS_FILENAME = "Aurexis_Knowledge_Base_Mock_Egypt.docx"

Kind = Literal["lookup", "multi_part", "comparison", "followup", "unanswerable"]
Modality = Literal["text", "table", "figure"]


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    #: Each entry must be satisfied. A tuple is satisfied by any one of its members.
    required: Sequence[str | tuple[str, ...]] = ()
    kind: Kind = "lookup"
    #: Where the answer lives in the corpus. `figure` cases are scored by modality.
    modality: Modality = "text"
    #: The turn before this one, for a follow-up. Retrieval is run on `question`, which is
    #: what query resolution is expected to produce from the two together.
    context: str = ""
    holdout: bool = False
    note: str = ""


CASES: list[Case] = [
    # ---------- tuition, from the fee table ----------
    Case("fees-y3-egyptian", "مصاريف Year 3 للطالب المصري كام؟",
         ["105,000 EGP", "Y03"], modality="table"),
    Case("fees-prek-international", "ابني في Pre-K وأجنبي، المصاريف كام؟",
         ["85,000 EGP", "Pre-K"], modality="table"),
    Case("fees-y11", "مصاريف Year 11 كام؟", ["150,000 EGP", "Y11"], modality="table"),
    Case("fees-fs1", "كام مصاريف FS1؟", ["88,000 EGP", "FS1"], modality="table", holdout=True),
    Case("fees-y7-both", "الفرق بين مصاريف المصري والدولي في Year 7 كام؟",
         ["120,000 EGP", "130,000 EGP"], kind="comparison", modality="table"),
    Case("fees-y9-international", "لو ابني دولي وفي Year 9 هدفع كام؟",
         ["145,000 EGP", "Y09"], modality="table", holdout=True),

    # ---------- discounts and payment policy ----------
    Case("sibling-discount", "في خصم لو عندي أكتر من ابن في المدرسة؟", ["7% discount"]),
    Case("referral-discount", "في خصم لو رشحت أهل صحابي للمدرسة؟", ["Referral Discount"], holdout=True),
    Case("re-enrollment-fee", "رسوم إعادة القيد كام وامتى آخر ميعاد؟",
         ["20% re-enrollment", "15th June"], kind="multi_part"),
    Case("withdrawal-notice", "لو عايز أسحب ابني من المدرسة لازم أبلغكم قبل بكام؟",
         ["One term in advance", ("Leaver", "admissions@aurexis.example")]),
    Case("withdrawal-no-notice", "لو مبلغتش بالانسحاب هيحصل إيه؟", ["Next term's fees will be due"]),
    Case("refund-illness", "لو ابني غاب بسبب مرض هيترد جزء من المصاريف؟",
         ["No refunds for absences"]),
    Case("refund-annual", "لو دفعت السنة كاملة ينفع أسترد فلوس الترم اللي مخدتوش؟",
         ["Eligible for refund", "full term's notice"], holdout=True),
    Case("outstanding-fees", "لو المصاريف اتأخرت هيحصل إيه؟",
         [("Withholding of academic reports", "Suspension or withdrawal of school place")]),
    Case("payment-bank", "أحول المصاريف على أنهي بنك؟", ["National Bank of Egypt"]),
    Case("payment-method", "إيه الطريقة المفضلة لدفع المصاريف؟", ["Online bank transfer"]),

    # ---------- admission ----------
    Case("admission-link", "أقدم لابني إزاي؟ فيه لينك؟", ["aurexis.example/admission"]),
    Case("admission-assessment", "فيه اختبار قبول؟ بيكون في إيه؟",
         [("Maths and English assessment", "assessment appointment")]),
    Case("active-grades", "بتقبلوا من أنهي سنة لأنهي سنة السنة دي؟", ["Pre-K to Year 8"]),
    Case("age-fs1", "ابني عنده 3 سنين، يدخل أنهي مرحلة؟", ["FS1", "3 years"]),
    Case("age-year2", "ابني عنده 6 سنين، يدخل أنهي سنة؟", ["Year 2", "6 years"], holdout=True),
    Case("accreditation", "المدرسة معتمدة من إيه؟", [("Cambridge Primary School", "Cambridge International Education")]),
    Case("admissions-contact", "أكلم مين في القبول؟", ["admissions@aurexis.example"]),

    # ---------- calendar ----------
    Case("term1-dates", "الترم الأول بيبدأ وبيخلص امتى؟", ["15/09", "18/12"]),
    Case("term2-start", "امتى يبدأ الترم التاني؟", ["11/01"]),
    Case("term3-dates", "مواعيد الترم التالت إيه؟", ["12/04", "25/06"], holdout=True),
    Case("school-since", "المدرسة شغالة من امتى؟", ["Sep 2024"]),

    # ---------- curriculum ----------
    Case("curriculum-only-british", "بتدرسوا المنهج الأمريكي ولا البريطاني؟",
         ["British curriculum"]),
    Case("british-vs-american", "إيه الفرق بين المنهج البريطاني والأمريكي؟",
         [("IGCSEs", "A-Levels")], kind="comparison"),
    Case("phonics-scheme", "بتدرسوا الفونكس بأنهي برنامج؟", ["Monster Phonics"]),
    Case("maths-programme", "منهج الرياضيات اسمه إيه؟", ["White Rose Maths"]),
    Case("subjects-early-years", "ابني في FS بيدرس إيه؟",
         [("Phonics", "Counting", "Arabic Language")], modality="table"),
    Case("subjects-year4", "ابني في Year 4 بيدرس أنهي مواد؟",
         [("Place Value", "Ancient Civilizations", "Coding Basics")], modality="table", holdout=True),
    Case("homework-frequency", "بيدوا واجبات كل قد إيه؟", ["homework each week"]),
    Case("egyptian-subjects", "بتدرسوا عربي ودين؟", [("Arabic Language", "Islamic")], modality="table"),

    # ---------- school life ----------
    Case("class-capacity", "الفصل فيه كام طالب؟", ["18 students"]),
    Case("clubs-list", "فيه أنشطة بعد المدرسة؟ زي إيه؟",
         [("Robotics", "Eco Club", "Football")]),
    Case("homework-club", "فيه مكان ابني يعمل فيه الواجب بعد المدرسة؟", ["Homework Clubs"], holdout=True),
    Case("behaviour-consequences", "بتتعاملوا إزاي لو الطفل سلوكه وحش؟",
         [("Reflection time", "Loss of certain privileges")]),
    Case("facilities-prayer", "فيه مكان للصلاة في المدرسة؟", ["Prayer Facilities"]),
    Case("transport-exists", "فيه باصات للمدرسة؟", [("school buses", "Transportation")]),

    # ---------- uniform: the answer is inside a photograph ----------
    Case("uniform-shop", "الزي المدرسي بشتريه منين؟", ["on-site school shop"]),
    Case("uniform-shoes", "الجزمة المدرسية لازم تكون شكلها إيه؟", ["plain black leather"]),
    Case("uniform-pe-trainers", "شروط كوتشي الرياضة إيه؟", ["non-marking soles"]),
    Case("uniform-eyfs-daywear", "زي الحضانة لبنتي بيتكون من إيه؟",
         modality="figure", note="EYFS day-wear photograph; transcription is vision output"),
    Case("uniform-primary-boys", "زي الولد في الابتدائي شكله إيه؟",
         modality="figure", note="day-wear photograph up to grade 6", holdout=True),

    # ---------- summer camp ----------
    Case("camp-dates", "الكامب الصيفي امتى؟", ["12 July", "6 August 2026"]),
    Case("camp-age", "الكامب من سن كام لسن كام؟", ["5 to 12 years"]),
    Case("camp-fullday-fee", "اشتراك اليوم الكامل في الكامب كام؟", ["4,000 EGP"], modality="table"),
    Case("camp-halfweek-fee", "برنامج نص الأسبوع في الكامب بكام؟", ["3,000 EGP"], modality="table", holdout=True),
    Case("camp-deadline", "آخر ميعاد للدفع في الكامب امتى؟", ["30 June 2026"]),
    Case("camp-registration", "أسجل في الكامب إزاي؟", ["summer-camp-2026"]),
    Case("camp-location", "الكامب هيكون فين؟", ["Fifth Settlement"]),

    # ---------- reporting and communication ----------
    Case("communication-channels", "بتتواصلوا مع الأهالي إزاي؟", [("ClassDojo", "Edunation")]),
    Case("assessment-methods", "بتقيموا الطلبة إزاي؟", [("Formative Assessment", "Summative Assessment")]),
    Case("progress-meetings", "فيه اجتماعات لمتابعة مستوى ابني؟",
         [("Pupil Progress Meetings", "Parent-Teacher meetings")]),
    Case("school-vision", "رؤية المدرسة إيه؟", ["Heart, Mind, Body and Soul"], holdout=True),

    # ---------- multi-part: two facts from two different sections ----------
    Case("fees-and-term", "مصاريف Year 6 كام والترم التاني بيبدأ امتى؟",
         ["105,000 EGP", "11/01"], kind="multi_part"),
    Case("camp-fee-and-deadline", "الكامب بكام للأسبوع الكامل وآخر ميعاد دفع امتى؟",
         ["4,000 EGP", "30 June 2026"], kind="multi_part"),

    # ---------- follow-ups: what resolution is expected to produce ----------
    Case("followup-fees-international", "طيب وللدولي؟",
         ["115,000 EGP", "Y03"], kind="followup", modality="table",
         context="مصاريف Year 3 للطالب المصري كام؟"),
    Case("followup-camp-halfday", "وبرنامج نص اليوم؟", ["2,500 EGP"], kind="followup",
         modality="table", context="اشتراك اليوم الكامل في الكامب كام؟"),

    # ---------- unanswerable: the corpus does not say ----------
    Case("bus-price", "اشتراك الباص بكام في الشهر؟", kind="unanswerable",
         note="the corpus says buses cost extra but never states a price"),
    Case("canteen-price", "أكل الكانتين بكام؟", kind="unanswerable",
         note="no canteen pricing in the corpus"),
    Case("other-branch", "فيه فرع للمدرسة في الإسكندرية؟", kind="unanswerable",
         note="no branch information in the corpus"),
    Case("teacher-salary", "مرتب المدرس عندكم كام؟", kind="unanswerable", holdout=True,
         note="not parent-facing material and not in the corpus"),

    # =====================================================================================
    # The wider set. Same rules, written the way a parent actually types: colloquial, often
    # indirect, sometimes two questions in one message, and never phrased as a lookup of the
    # document's own wording. The corpus answers them; nothing here is shaped around where a
    # chunk happens to begin or end.
    # =====================================================================================

    # ---------- getting in: steps, documents, assessment ----------
    Case("apply-how", "عايزة أقدم لبنتي، أبدأ منين؟", ["aurexis.example/admission"]),
    Case("apply-visit-school", "ينفع أزور المدرسة وأشوفها قبل ما أقدم؟",
         [("tour the facilities", "school tour")]),
    Case("apply-age-cutoff", "بتحددوا السنة الدراسية لابني على أساس إيه؟", ["age of the child in October"]),
    Case("apply-after-assessment", "بعد الاختبار بيحصل إيه لو ابني اتقبل؟", ["offer will be sent"]),
    Case("apply-assessment-content", "الاختبار اللي بتعملوه للطفل بيبقى في إيه؟",
         ["Maths and English assessment"]),
    Case("docs-basic", "محتاجة أجهز أنهي ورق عشان التقديم؟",
         [("Birth Certificate", "Vaccination Certificate")], kind="multi_part"),
    Case("docs-transfer-inside", "ابني منقول من مدرسة في مصر، محتاج ورق زيادة؟",
         [("Financial clearance", "MOE-stamped")]),
    Case("docs-transfer-abroad", "إحنا جايين من برة مصر، الشهادات بتاعته لازم تتعمل لها إيه؟",
         ["embassy"], holdout=True),
    Case("docs-vaccination", "شهادة التطعيمات مطلوبة ولا لأ؟", ["Vaccination Certificate"]),
    Case("accreditation-id", "المدرسة مسجلة برقم كام في كامبريدج؟", ["EG-MOCK-001"], holdout=True),
    Case("accreditation-date", "اعتماد كامبريدج اتاخد امتى؟", ["15 September 2025"]),
    Case("grades-open-now", "بتقبلوا دلوقتي لحد أنهي سنة دراسية؟",
         [("Foundation Stage to Year 5", "Pre-K to Year 8")]),
    Case("grades-y7-next-year", "ابني هيبقى في Year 7 السنة الجاية، ينفع أحجزله؟", ["Y6 to Y8"]),
    Case("age-under-three", "بنتي لسه عندها سنتين ونص، تنفع تدخل عندكم؟",
         [("Pre-K", "under 3")]),
    Case("age-year5", "ابني عنده 9 سنين هيتحط في أنهي سنة؟", ["Year 5", "9 years"], holdout=True),
    Case("egypt-equivalent-kg2", "FS2 دي بتقابل إيه في النظام المصري؟", ["KG2"]),
    Case("egypt-equivalent-grade5", "Year 6 عندكم يعني الصف الخامس الابتدائي؟", ["Grade 5"]),

    # ---------- money: the part parents ask about most ----------
    Case("fees-what-excluded", "المصاريف اللي بتقولوا عليها شاملة إيه بالظبط؟",
         [("Excluded from tuition fees", "Books")]),
    Case("fees-books-included", "الكتب داخلة في المصاريف ولا هدفع عليها لوحدها؟", ["Books"]),
    Case("fees-currency", "الدفع بالدولار ولا بالمصري؟", ["charged in EGP"]),
    Case("fees-increase-notice", "لو المصاريف هتزيد بتقولوا لنا قبلها بقد إيه؟",
         ["half a term in advance"]),
    Case("fees-moe-approved", "المصاريف دي متوافق عليها من الوزارة؟", ["Ministry of Education"]),
    Case("payment-installments", "ينفع أقسط المصاريف ولا لازم تتدفع مرة واحدة؟",
         [("flexible payment plans", "Down Payment")]),
    Case("payment-first-due", "أول قسط بيتدفع امتى؟", ["1st August"]),
    Case("payment-term2-due", "القسط اللي بعد كده ميعاده امتى؟", ["1st November"]),
    Case("payment-term3-due", "آخر قسط في السنة بيبقى امتى؟", ["1st March"], holdout=True),
    Case("payment-downpayment-share", "المقدم اللي بدفعه عند القبول بيبقى كام في المية؟", ["Down Payment (20%)"]),
    Case("payment-example-fs1", "لو ابني FS1 المقدم هيطلع كام بالجنيه؟",
         ["17,600 EGP"], kind="multi_part"),
    Case("payment-joining-term2", "لو ابني هيدخل من الترم التاني هدفع إزاي؟", ["40% for Term 2"]),
    Case("payment-cards", "بتقبلوا الدفع بالفيزا؟", [("Visa", "Meeza")]),
    Case("payment-cheque", "ينفع أدفع بشيك؟", ["cheques, are not accepted"]),
    Case("payment-proof", "أبعت إيصال التحويل على إيه؟", ["finance@aurexis.example"]),
    Case("payment-hardship", "ظروفي المادية اتغيرت ومش هقدر أدفع في الميعاد، أعمل إيه؟",
         [("Finance Department", "before the due date")]),
    Case("marketplace", "الكتب والباص بدفعهم فين؟", ["online marketplace"], holdout=True),
    Case("meals-available", "بتوفروا أكل في المدرسة؟", [("external caterer", "hot, nutritious meals")]),
    Case("meals-optional", "لازم أشترك في وجبات المدرسة؟", [("optional", "paid per semester")]),
    Case("late-enrollment-discount", "لو ابني دخل متأخر شهر عن بداية الترم هدفع الترم كامل؟",
         ["one month's tuition"]),
    Case("early-payment-discount", "لو دفعت السنة كلها مقدم في خصم؟",
         ["not confirmed"], note="the corpus explicitly says this is unconfirmed"),
    Case("corporate-discount", "شركتي متعاقدة معاكم، في خصم لموظفيها؟",
         [("Corporate Partnership", "20% tuition fee reduction")]),
    Case("corporate-proof", "عشان آخد خصم الشركة محتاج أجيب إيه؟",
         [("HR letter", "company ID")]),
    Case("corporate-joined-later", "اشتغلت في شركة متعاقدة معاكم بعد ما ابني دخل، الخصم هيتطبق؟",
         ["following semester"], holdout=True),
    Case("corporate-contact", "أكلم مين بخصوص اتفاقية الشركات؟",
         [("Mariam Hassan", "mariam.hassan@aurexis.example")]),
    Case("corporate-partners", "شركة NileTech من ضمن الشركات المتعاقدة؟", ["NileTech"]),
    Case("corporate-with-sibling", "خصم الإخوات بيتجمع مع خصم الشركة؟",
         ["in addition to corporate discounts"], kind="comparison"),
    Case("founders-fees", "إحنا من أول سنة في المدرسة، ليّا وضع خاص في المصاريف؟",
         ["Founders Fees"]),
    Case("founders-duration", "سعر المؤسسين ده بيفضل معايا لحد امتى؟",
         ["full educational journey"], holdout=True),

    # ---------- what the child actually studies ----------
    Case("curriculum-which", "المدرسة بتدرس أنهي نظام؟", [("UK national curriculum", "British")]),
    Case("curriculum-books", "بتستخدموا كتب إيه؟", ["Cambridge textbooks"]),
    Case("curriculum-igcse", "ابني هيطلع منكم بشهادة إيه؟", ["IGCSEs"]),
    Case("curriculum-american-option", "لو حبيت المنهج الأمريكي بدل البريطاني ينفع؟",
         ["not any other curriculum"]),
    Case("curriculum-choose-help", "محتار بين البريطاني والأمريكي، أقدر أستشير حد عندكم؟",
         ["principal"], kind="comparison"),
    Case("eyfs-approach", "بنتي في الحضانة بتتعلم إزاي؟ بيلعبوا بس؟", ["learning through play"]),
    Case("eyfs-areas-count", "منهج الحضانة بيغطي كام مجال؟", ["seven areas"]),
    Case("eyfs-prime-areas", "إيه أهم حاجات بيركزوا عليها في الحضانة؟",
         [("Communication and Language", "Personal, Social and Emotional Development")]),
    Case("eyfs-principles", "فلسفتكم في التعامل مع أطفال الحضانة إيه؟",
         [("unique child", "positive relationships")], holdout=True),
    Case("ks1-description", "Year 1 و Year 2 بيبقى شكل الدراسة إزاي؟",
         [("More structured learning", "Cambridge International")]),
    Case("arabic-mandatory", "العربي والدين بيتدرسوا عندكم ولا مدرسة أجنبية بس؟",
         [("Arabic", "Islamic")]),
    Case("non-muslim-students", "إحنا مش مسلمين، ابني هيعمل إيه وقت حصة الدين؟",
         ["Topic lessons"]),
    Case("egyptian-identity", "ابني هيفضل متمسك بهويته المصرية في مدرسة بريطانية؟",
         [("Egyptian social studies", "pride in identity")]),
    Case("french-language", "بتدرسوا لغة تانية غير الإنجليزي؟", ["French"]),
    Case("swimming", "فيه حصص سباحة؟", ["Swimming"], modality="table", holdout=True),
    Case("science-scheme", "منهج العلوم بتاعكم اسمه إيه؟", ["White Rose Science"]),
    Case("english-scheme", "حصص الإنجليزي بتتدرس بأنهي طريقة؟", ["Literacy Tree"]),
    Case("phonics-accredited", "برنامج الفونكس معتمد من جهة رسمية؟",
         ["Department of Education England"]),
    Case("homework-early-years", "بنتي في الحضانة بياخدوا واجبات؟", ["key words and sounds"]),
    Case("homework-year3", "ابني في Year 3 بياخد واجب في إيه؟",
         [("Maths and English", "spellings")]),
    Case("computing", "بيتعلموا برمجة ولا كمبيوتر عادي؟",
         [("Coding Basics", "Digital Research")], modality="table"),
    Case("stem-approach", "بتعملوا حاجة اسمها STEM؟ دي إيه؟",
         [("Interdisciplinary", "project learning")], modality="table", holdout=True),
    Case("pshe", "بتعلموهم حاجات زي الثقة بالنفس والقيادة؟",
         [("Leadership Skills", "Resilience")], modality="table"),

    # ---------- progress, staff and support ----------
    Case("assessment-daily", "بتعرفوا مستوى ابني إزاي من غير امتحانات كتير؟",
         ["Formative Assessment"]),
    Case("assessment-formal", "فيه امتحانات رسمية ولا لأ؟", ["Summative Assessment"]),
    Case("cat4", "فيه اختبارات مستوى بتتعمل لكل الطلبة؟", [("CAT4", "GL Assessment")]),
    Case("cat4-years", "اختبار الـ CAT4 بيتعمل في أنهي سنين؟", ["Years 4, 7 and 9"], holdout=True),
    Case("reports-when", "بيجيلي تقرير عن ابني كام مرة في السنة؟",
         [("Terms 1", "full report")]),
    Case("report-term3", "آخر السنة بيجيلي إيه عن مستوى ابني؟",
         [("overall summary", "summer revision")]),
    Case("parent-meetings", "بقابل مدرس ابني امتى؟", ["Parent-Teacher meetings"]),
    Case("ppm-timing", "بتراجعوا مستوى الطلبة كل قد إيه جوه المدرسة؟",
         ["midpoint of each half-term"]),
    Case("teachers-nationality", "المدرسين عندكم مصريين ولا أجانب؟",
         [("British nationals", "UK-recognised")]),
    Case("specialist-teachers", "مين بيدي حصص الموسيقى والرياضة؟",
         [("Specialist Staff", "PE, Music, Art")]),
    Case("pastoral-contact", "لو ابني عنده مشكلة في المدرسة أكلم مين الأول؟",
         [("Class Teachers", "Form Tutors")]),
    Case("nurse", "فيه دكتور أو ممرضة في المدرسة؟", ["School Nurse"]),
    Case("safeguarding", "إيه اللي بيأمن ابني جوه المدرسة؟", ["Safeguarding"], holdout=True),
    Case("senco", "ابني محتاج دعم إضافي في التعلم، بتوفروا كده؟", [("SENCo", "Inclusion")]),
    Case("english-support", "ابني إنجليزيه ضعيف، هيتظبط إزاي؟", ["English language support"]),
    Case("library", "فيه مكتبة للأطفال؟", [("Library", "Librarian")]),

    # ---------- the school day, behaviour, and the building ----------
    Case("school-hours", "اليوم الدراسي بيبدأ وبينتهي الساعة كام؟", ["7:45 AM"]),
    Case("weekend", "المدرسة بتقفل يوم إيه؟", ["Closed on Fridays"]),
    Case("afternoon-activities", "فيه حاجة للأطفال بعد اليوم الدراسي؟",
         ["After-school activities"]),
    Case("behaviour-philosophy", "بتربوا الأطفال على الانضباط إزاي؟",
         [("Teaching Positive Choices", "Taking Responsibility")]),
    Case("behaviour-parents-involved", "لو ابني عمل مشكلة متكررة هتكلموني؟",
         ["Family partnership"], holdout=True),
    Case("facilities-outdoor", "فيه مكان للأطفال يلعبوا فيه برة؟", ["Outdoor Learning"]),
    Case("facilities-specialist-rooms", "فيه فصول مخصصة للفن والموسيقى؟", ["Specialist Learning Rooms"]),
    Case("wellness-space", "لو ابني حاسس بضغط نفسي فيه حد يساعده؟", ["Wellness"]),
    Case("school-features", "إيه اللي يميز مدرستكم عن غيرها؟",
         [("Qualified British Staff", "Exceptional Facilities")]),
    Case("parent-role", "المدرسة عايزة إيه من الأهالي؟", ["active participation"]),

    # ---------- uniform: the answers that live in photographs ----------
    Case("uniform-buy-where", "الزي بشتريه من برة ولا من المدرسة؟", ["on-site school shop"]),
    Case("uniform-pe-color", "كوتشي الرياضة ينفع يكون أسود؟", ["not BLACK"]),
    Case("uniform-shoes-details", "الجزمة لازم تكون كعب ولا فلات؟", ["flat-heeled"]),
    Case("uniform-velcro", "ابني صغير ومش عارف يربط الرباط، فيه حل؟", ["Velcro"], holdout=True),
    Case("uniform-trainers-banned", "ينفع يلبس كوتشي عادي بدل الجزمة؟", ["not permitted"]),
    Case("uniform-secondary-look", "زي الثانوي شكله إيه؟", modality="figure",
         note="secondary day-wear photograph"),
    Case("uniform-pe-kit-look", "بدلة الرياضة شكلها إيه؟", modality="figure",
         note="PE kit photograph, unisex"),
    Case("uniform-girls-primary", "بنتي في Year 4، الزي بتاعها بيتكون من إيه؟",
         modality="figure", holdout=True, note="girls day-wear up to grade 6"),

    # ---------- calendar, transport, and getting there ----------
    Case("year-end", "السنة الدراسية بتخلص امتى؟", ["02nd July"]),
    Case("term2-end", "الترم التاني بيخلص امتى؟", ["26/03"]),
    Case("holidays-where", "أعرف منين مواعيد الإجازات؟", ["academic-calender"]),
    Case("bus-districts", "إحنا ساكنين في مدينتي، الباص بيوصل عندنا؟", ["Madinaty"]),
    Case("bus-maadi", "الباص بيجي المعادي؟", ["Maadi"], holdout=True),
    Case("bus-coverage-list", "الباص بيغطي أنهي مناطق؟",
         [("New Cairo", "Nasr City", "Heliopolis")]),
    Case("bus-updates", "أعرف منين لو المناطق اتغيرت؟", ["transportation"]),
    Case("school-address", "المدرسة عنوانها فين بالظبط؟",
         [("Fifth Settlement", "New Cairo")]),
    Case("school-map", "ابعتلي لوكيشن المدرسة", ["maps.aurexis.example"]),
    Case("school-branches", "فيه فروع تانية للمدرسة؟", ["first branch"]),
    Case("phone-number", "فيه رقم تليفون أكلمكم عليه؟", ["100 000 0000"]),
    Case("whatsapp", "ينفع أكلمكم واتساب؟", ["WhatsApp"], holdout=True),
    Case("admissions-hours", "بترودوا على الاستفسارات من الساعة كام؟", ["7am-3:30 pm"]),
    Case("careers", "عايزة أشتغل عندكم، أبعت السي في على إيه؟", ["careers@aurexis.example"]),

    # ---------- camp and clubs ----------
    Case("camp-what-is", "الكامب الصيفي بيعملوا فيه إيه؟",
         [("STEAM", "robotics")]),
    Case("camp-halfday-fee2", "لو عايزة أسجلها نص يوم بس بكام؟", ["2,500 EGP"], modality="table"),
    Case("camp-activities-coding", "بيتعلموا برمجة في الكامب؟", ["Coding"]),
    Case("camp-age-limit", "ابني عنده 4 سنين ينفع يحضر الكامب؟", ["5 to 12 years"]),
    Case("camp-host", "الكامب بيتعمل جوه المدرسة ولا برة؟", [("Hosted at Aurexis", "Aurexis")]),
    Case("clubs-sports", "فيه كورة أو باسكت بعد المدرسة؟", [("Football", "Basketball")]),
    Case("clubs-stem", "فيه نادي روبوتيكس؟", ["Robotics"]),
    Case("clubs-arts", "بنتي بتحب الرسم والتمثيل، فيه حاجة ليها؟",
         [("Art & Design", "Drama")], holdout=True),
    Case("clubs-leadership", "فيه أنشطة بتعلمهم القيادة والخطابة؟",
         [("Debate", "Kindness Ambassadors")]),

    # ---------- communication ----------
    Case("comms-app", "بتبعتوا أخبار ابني على أنهي تطبيق؟", ["ClassDojo"]),
    Case("comms-system", "المنصة الرسمية للمدرسة اسمها إيه؟", ["Edunation"]),
    Case("comms-arabic", "أنا مش بتكلم إنجليزي كويس، هتتعاملوا معايا إزاي؟",
         [("Bilingual", "Arabic-speaking")]),
    Case("comms-newsletter", "بيجيلي حاجة أسبوعية عن أخبار المدرسة؟", ["Newsletter"], holdout=True),
    Case("comms-complaint", "عندي شكوى، أعملها إزاي؟", ["speak up early"]),

    # ---------- follow-ups: the second message in a conversation ----------
    Case("followup-fees-y7", "طيب و Year 7؟", ["120,000 EGP"], kind="followup",
         modality="table", context="مصاريف Year 3 للطالب المصري كام؟"),
    Case("followup-docs-egyptian", "وإحنا مصريين، في ورق زيادة؟", ["Family Record"],
         kind="followup", context="محتاجة أجهز أنهي ورق عشان التقديم؟"),
    Case("followup-bus-area", "وبيجي التجمع؟", ["Fifth Settlement"], kind="followup",
         context="فيه باصات للمدرسة؟"),
    Case("followup-camp-age", "وبنتي عندها 6 سنين تنفع؟", ["5 to 12 years"], kind="followup",
         context="الكامب الصيفي امتى؟", holdout=True),
    Case("followup-uniform-pe", "وبدلة الرياضة؟", modality="figure", kind="followup",
         context="زي الولد في الابتدائي شكله إيه؟"),
    Case("followup-refund", "وياترى لو سحبته بعد ما دفعت السنة؟",
         [("Eligible for refund", "full term's notice")], kind="followup",
         context="لو عايز أسحب ابني من المدرسة لازم أبلغكم قبل بكام؟"),
    Case("followup-assessment-years", "وبيتعاد في أنهي سنين؟", ["Years 4, 7 and 9"],
         kind="followup", context="فيه اختبارات مستوى بتتعمل لكل الطلبة؟"),
    Case("followup-discount-second", "وللتالت كمان؟", ["7% discount"], kind="followup",
         context="في خصم لو عندي أكتر من ابن في المدرسة؟"),

    # ---------- two questions in one message ----------
    Case("fees-and-docs", "مصاريف FS1 كام ومحتاج أجهز أنهي ورق؟",
         ["88,000 EGP", ("Birth Certificate", "Vaccination Certificate")], kind="multi_part"),
    Case("hours-and-bus", "اليوم الدراسي بيخلص الساعة كام والباص بيغطي التجمع؟",
         ["7:45 AM", "Fifth Settlement"], kind="multi_part"),
    Case("camp-and-clubs", "الكامب بيبدأ امتى وفيه أنشطة بعد المدرسة طول السنة؟",
         ["12 July", ("Robotics", "Football")], kind="multi_part", holdout=True),
    Case("reenroll-and-discount", "رسوم إعادة القيد كام ولو عندي ولدين في خصم؟",
         ["20% re-enrollment", "7% discount"], kind="multi_part"),
    Case("curriculum-and-arabic", "بتدرسوا بريطاني وكمان عربي ودين؟",
         [("British", "UK national curriculum"), ("Arabic", "Islamic")], kind="multi_part"),

    # ---------- comparisons the parent makes, not the document ----------
    Case("compare-fees-y1-y3", "الفرق في المصاريف بين Year 1 و Year 3 كام؟",
         ["95,000 EGP", "105,000 EGP"], kind="comparison", modality="table"),
    Case("compare-camp-programmes", "أنهي أوفر ليا، نص يوم ولا نص أسبوع؟",
         ["2,500 EGP", "3,000 EGP"], kind="comparison", modality="table"),
    Case("compare-egyptian-international-prek", "ابني مصري وصاحبه أجنبي، هندفع نفس المصاريف في Pre-K؟",
         ["75,000 EGP", "85,000 EGP"], kind="comparison", modality="table", holdout=True),
    Case("compare-corporate-vs-sibling", "خصم الشركة أكبر ولا خصم الإخوات؟",
         ["20% tuition fee reduction", "7% discount"], kind="comparison"),

    # ---------- the corpus says it does not know ----------
    Case("workbook-fees", "الكتب والكشاكيل بكام؟", kind="unanswerable",
         note="the corpus lists workbook and stationery fees as information it does not have"),
    Case("books-refundable", "لو ابني سحب، فلوس الكتب بترجع؟", kind="unanswerable",
         note="listed in the corpus as not available"),
    Case("book-list", "ممكن قائمة الكتب المطلوبة للصف التالت؟", kind="unanswerable",
         note="listed in the corpus as not available"),
    Case("uniform-price", "الزي المدرسي بكام؟", kind="unanswerable", holdout=True,
         note="the shop is named but no price is given"),
    Case("exam-results", "نتيجة ابني في الامتحان طلعت؟", kind="unanswerable",
         note="a records question, not a knowledge-base one"),
    Case("teacher-name", "مين مدرس ابني السنة دي؟", kind="unanswerable",
         note="a records question, not a knowledge-base one"),
    Case("weather-question", "الجو هيبقى عامل إيه بكرة؟", kind="unanswerable",
         note="plainly outside the assistant's subject"),
]


def cases(split: str = "dev") -> list[Case]:
    """`dev` (default), `holdout`, or `all`."""
    if split == "all":
        return list(CASES)
    if split == "holdout":
        return [c for c in CASES if c.holdout]
    if split == "dev":
        return [c for c in CASES if not c.holdout]
    raise ValueError(f"unknown split {split!r}; use dev, holdout, or all")


def satisfied(requirement: str | tuple[str, ...], haystack: str) -> bool:
    """Whether one requirement is met by `haystack`, case-insensitively."""
    text = haystack.lower()
    if isinstance(requirement, tuple):
        return any(item.lower() in text for item in requirement)
    return requirement.lower() in text


def missing(case: Case, haystack: str) -> list[str | tuple[str, ...]]:
    """The requirements `haystack` does not satisfy, in declaration order."""
    return [item for item in case.required if not satisfied(item, haystack)]
