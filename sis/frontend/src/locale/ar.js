/*
 * The Arabic console, as a list of sentences.
 *
 * Keyed by the English source string, so this file can be read end to end by somebody who does
 * not read the code, and a line that no longer appears anywhere is a line that can be deleted
 * without hunting for its call site. A string missing from here renders in English rather than
 * as a key or a blank — see the note in `i18n.js` about why that is the right failure.
 *
 * -- House rules for the Arabic ------------------------------------------------------
 *
 * `{0}` marks a hole and must survive translation, in whatever position Arabic puts it. It is
 * the class code, the count or the link that the sentence is about, and a sentence that loses
 * its hole renders with the value missing.
 *
 * Latin identifiers stay Latin. A class is 3A in both languages, a student number is Latin
 * digits in both, and a year is 2025-2026 in both — those are the school's own identifiers,
 * not words, and transliterating them would mean a registrar could not match what is on the
 * screen against what is on the paper in their hand.
 *
 * Terms of art follow what Egyptian school administration actually calls them rather than the
 * dictionary: مرحلة for a stage, صف for a class section, ولي أمر for a guardian, كشف for a
 * register. The English "rung" is this system's own coinage for a year-level; it is صف دراسي.
 */
export const AR = {
  /* -- The shell ----------------------------------------------------------------- */
  'Student Information Service': 'نظام معلومات الطلاب',
  'Registrar console': 'وحدة تحكم شؤون الطلاب',
  School: 'المدرسة',
  'Find a child': 'البحث عن طالب',
  Roster: 'كشف القيد',
  Guardians: 'أولياء الأمور',
  Marks: 'الدرجات',
  Batches: 'دفعات الرفع',
  Year: 'العام',
  Schools: 'المدارس',
  'Open settings': 'فتح الإعدادات',
  'Settings — appearance, page colour, names': 'الإعدادات — المظهر ولون الصفحة والأسماء',
  Screens: 'الشاشات',
  'service online': 'الخدمة متاحة',
  'service unreachable': 'تعذّر الوصول إلى الخدمة',
  'checking…': 'جارٍ الفحص…',
  'Marks are stated figures, reported exactly as the school recorded them. A blank is not a zero.':
    'الدرجات أرقام مُصرَّح بها، تُعرض كما سجّلتها المدرسة تمامًا. والخانة الفارغة ليست صفرًا.',
  'API reference': 'مرجع الواجهة البرمجية',

  /* -- Settings ------------------------------------------------------------------ */
  Settings: 'الإعدادات',
  'Close settings': 'إغلاق الإعدادات',
  Appearance: 'المظهر',
  Light: 'فاتح',
  Dark: 'داكن',
  System: 'حسب الجهاز',
  'System follows the machine, and changes with it while the tab is open.':
    'يتبع «حسب الجهاز» إعداد الجهاز نفسه، ويتغيّر معه ما دامت الصفحة مفتوحة.',
  'Page colour': 'لون الصفحة',
  'The colour of the page. Every section, hairline and field on it is mixed from this one value, so the whole console follows.':
    'لون الصفحة. كل قسم وخط فاصل وحقل عليها مشتق من هذه القيمة وحدها، فتتبعها الواجهة كاملة.',
  'The colour of the dark page. The light one keeps its own, and switching appearance switches between them.':
    'لون الصفحة الداكنة. وللصفحة الفاتحة لونها الخاص، والتبديل بين المظهرين يبدّل بينهما.',
  'Any colour': 'أي لون',
  'Pick any page colour': 'اختر أي لون للصفحة',
  'Reset to default': 'العودة إلى الأصل',
  Names: 'الأسماء',
  'Which of a child’s two recorded names is shown first. The console’s own wording stays in English — a half-translated interface is worse than an untranslated one.':
    'أيّ الاسمين المسجَّلين للطالب يظهر أولًا، ولغة الواجهة نفسها.',
  'Latin names first': 'الأسماء اللاتينية أولًا',
  'Arabic names first, right to left': 'الأسماء العربية أولًا، من اليمين إلى اليسار',
  'Remembered in this browser. Nothing here is sent to the service.':
    'تُحفظ هذه الإعدادات في هذا المتصفح، ولا يُرسل منها شيء إلى الخدمة.',
  Done: 'تم',
  'The page, a section on it, a hairline, a field and the one blue — the same tokens every screen is drawn from, at a quarter size.':
    'الصفحة، وقسم عليها، وخط فاصل، وحقل، واللون الأزرق الوحيد — بالقيم نفسها التي تُرسم بها كل شاشة، بربع الحجم.',
  Paper: 'ورقي',
  White: 'أبيض',
  Mist: 'ضبابي',
  Cream: 'كريمي',
  Sand: 'رملي',
  Charcoal: 'فحمي',
  Black: 'أسود',
  Ink: 'حبري',
  Umber: 'بنّي',
  Steel: 'فولاذي',

  /* -- Fields -------------------------------------------------------------------- */
  'Choose…': 'اختر…',
  'Filter…': 'تصفية…',
  'Filter the list': 'تصفية القائمة',
  'Nothing matches': 'لا توجد نتائج مطابقة',
  Clear: 'مسح',

  /* -- The screens ---------------------------------------------------------------
   *
   * Everything below is the console's own wording, extracted from the nine screens and the
   * components they share. Sorted by the English rather than grouped by screen: a sentence
   * usually appears on more than one, and filing it under the first screen somebody noticed
   * it on is how a table grows two entries for one string.
   */
  '1. The file':
    'الملف',
  'A child nobody has marked shows':
    'الطالب الذي لم يسجّل له أحد شيئًا يظهر',
  'A child whose record exists already — a transfer in, or one uploaded but never placed.':
    'طالب سجلّه موجود بالفعل — محوّل إلينا، أو رُفع دون أن يُلحق بصف.',
  "A marks upload names a subject from this year's catalogue, so nothing can be recorded until one exists.":
    'رفع الدرجات يستدعي مادة من كتالوج هذا العام، فلا يمكن تسجيل شيء قبل وجود مادة.',
  'A partial name matches in either script.':
    'يكفي جزء من الاسم بأي من الكتابتين.',
  "A person marked this term final, so the service will refuse the upload. Reopen the term from the academic year screen if last term's marks are genuinely still arriving.":
    'اعتمد أحدهم هذا الفصل الدراسي نهائيًا، ولذلك سترفض الخدمة الرفع. أعد فتح الفصل من شاشة العام الدراسي إن كانت درجات الفصل الماضي ما زالت تصل فعلًا.',
  'A placement is a dated membership, so a transfer closes one row and opens another. October still says what it said in October — nothing here is rewritten when she moves.':
    'الإلحاق عضوية مؤرَّخة، فالنقل يغلق سطرًا ويفتح آخر. ويبقى أكتوبر قائلًا ما قاله في أكتوبر — لا يُعاد كتابة شيء هنا عند انتقالها.',
  "A rung can only be removed while nothing points at it. This school's classes on":
    'لا يُحذف الصف الدراسي إلا وليس عليه أي مرجع. وصفوف هذه المدرسة على',
  'A subject belongs to this year. The same code in another year is a different subject, so marks are not comparable across years — copy the catalogue forward each September rather than expecting it to carry over.':
    'المادة تخصّ عامها. والرمز نفسه في عام آخر مادة أخرى، فالدرجات غير قابلة للمقارنة بين الأعوام — انسخ الكتالوج إلى العام التالي كل سبتمبر بدلًا من انتظار انتقاله تلقائيًا.',
  'Absent':
    'غائب',
  'Academic year':
    'العام الدراسي',
  'Academic years':
    'الأعوام الدراسية',
  'Add academic year':
    'إضافة عام دراسي',
  'Add one above, or open an academic year and use the generator to build the whole ladder at once.':
    'أضف واحدًا بالأعلى، أو افتح عامًا دراسيًا واستخدم المولّد لبناء السلّم كاملًا دفعة واحدة.',
  "Add one above, or open the academic year and generate the whole ladder's classes at once.":
    'أضف واحدًا بالأعلى، أو افتح العام الدراسي وولّد صفوف السلّم كلها دفعة واحدة.',
  'Add rung':
    'إضافة صف دراسي',
  'Add school':
    'إضافة مدرسة',
  'Add term':
    'إضافة فصل دراسي',
  'Add the first child':
    'إضافة أول طالب',
  'Add the first year':
    'إضافة أول عام',
  "As stored. A national number is matched against the school's dialling code.":
    'كما هو مسجَّل. ويُطابَق الرقم الوطني مع رمز الاتصال الخاص بالمدرسة.',
  'Attendance':
    'الحضور',
  'Batch id':
    'رقم الدفعة',
  'Builds':
    'يبني',
  'By student number, or by part of a name in either script.':
    'برقم الطالب، أو بجزء من الاسم بأي من الكتابتين.',
  'Cancel':
    'إلغاء',
  'Capacity':
    'السعة',
  'Child':
    'الطالب',
  'Choose a year first':
    'اختر عامًا أولًا',
  'Choose the guardians sheet':
    'اختر ملف أولياء الأمور',
  'Choose the marks sheet':
    'اختر ملف الدرجات',
  'Choose the roster sheet':
    'اختر ملف كشف القيد',
  'Choose the term first. A marks file with no term named is a file the service cannot place.':
    'اختر الفصل الدراسي أولًا. فملف درجات بلا فصل مُسمّى ملف لا تستطيع الخدمة تحديد موضعه.',
  'Chosen in the header, for every screen.':
    'مختار من الشريط العلوي، ولكل الشاشات.',
  'Class':
    'الصف',
  'Class code':
    'رمز الصف',
  'Class this term':
    'الصف في هذا الفصل',
  'Classes':
    'الصفوف',
  'Clear the filter to see the rest.':
    'امسح التصفية لعرض الباقي.',
  'Code':
    'الرمز',
  'Codes are unique within an academic year.':
    'الرموز فريدة داخل العام الدراسي الواحد.',
  'Comma separated; the first n are used.':
    'مفصولة بفواصل؛ يُؤخذ أول ن منها.',
  'Contact phone':
    'هاتف التواصل',
  'Create the academic year':
    'إنشاء العام الدراسي',
  'Create the first school':
    'إنشاء أول مدرسة',
  'Date of birth':
    'تاريخ الميلاد',
  'Discard changes':
    'إلغاء التغييرات',
  'Dismiss':
    'إخفاء',
  'Division':
    'المرحلة',
  "Double-click a row to open a child's record — or tap her name, which is the same place and works on a phone.":
    'افتح سجل الطالب بالنقر على صفّه — أو انقر اسمها، وهو المكان نفسه ويعمل على الهاتف.',
  'Either no marks have been uploaded for this term, or she has no placement in it.':
    'إمّا أنه لم تُرفع درجات لهذا الفصل، وإمّا أنها غير ملحقة بأي صف فيه.',
  'Either the number is not on file, or every child it is linked to is restricted from it.':
    'إمّا أن الرقم غير مسجَّل، وإمّا أن كل طفل مرتبط به ممنوع عنه الاطّلاع.',
  'Either the year has no subjects yet, or nothing has been uploaded against this term.':
    'إمّا أن العام بلا مواد بعد، وإمّا أنه لم يُرفع شيء على هذا الفصل.',
  'Enrol children and place them in classes. A placement is a dated membership: a child who moves from 3A to 3B in March is in 3A for Term 1 and 3B for Term 2, and both stay true.':
    'قيّد الطلاب وألحقهم بالصفوف. الإلحاق عضوية مؤرَّخة: الطالبة التي تنتقل من 3A إلى 3B في مارس هي في 3A للفصل الأول وفي 3B للفصل الثاني، وكلاهما صحيح.',
  "Every mark already stated against it stays exactly as it is — retiring is not deleting, and this year's report cards keep their heading.":
    'كل درجة صُرِّح بها عليها تبقى كما هي تمامًا — التقاعد ليس حذفًا، وشهادات هذا العام تحتفظ بعنوانها.',
  'Every subject named in the file':
    'كل مادة مُسمّاة في الملف',
  'Excused':
    'بعذر',
  'Field at fault:':
    'الحقل محل الخطأ:',
  'Figures exactly as the school stated them, in subject order. Nothing here is averaged, weighted or ranked, and a dash is a subject awaiting a mark — never a zero.':
    'أرقام كما صرّحت بها المدرسة تمامًا، بترتيب المواد. لا شيء هنا يُحسب متوسطه أو يُرجَّح أو يُرتَّب، والشرطة مادة تنتظر درجتها — وليست صفرًا أبدًا.',
  'Fills only the children still blank, and leaves every mark already on file alone.':
    'يملأ الطلاب الذين ما زالت خانتهم فارغة فقط، ويترك كل درجة مسجَّلة كما هي.',
  'Find a batch':
    'البحث عن دفعة',
  'First day':
    'أول يوم',
  'From':
    'من',
  'Generate the ladder':
    'توليد السلّم',
  'Generating…':
    'جارٍ التوليد…',
  'Go back to the school':
    'العودة إلى المدرسة',
  'Go to the school':
    'الانتقال إلى المدرسة',
  'Grouped by division, youngest first. A division is a label for reading a long ladder — no rule anywhere depends on it, so moving a rung between divisions changes which heading it sits under and nothing else.':
    'مجمَّعة حسب المرحلة، الأصغر أولًا. والمرحلة تسمية تُيسِّر قراءة سلّم طويل — لا تعتمد عليها أي قاعدة، فنقل صف دراسي بين المراحل يغيّر العنوان الذي يقع تحته ولا شيء غير ذلك.',
  'Her record exists and no class has claimed her. Place her from a class register.':
    'سجلّها موجود ولم يطالب بها أي صف. ألحقها من كشف صف.',
  'How to show the classes':
    'طريقة عرض الصفوف',
  'If she is moving to another class, use':
    'إن كانت تنتقل إلى صف آخر، فاستخدم',
  'Immutable. Every year and rung in the school points at it.':
    'غير قابل للتغيير. كل عام وصف دراسي في المدرسة يشير إليه.',
  'In the':
    'في',
  'In the class from':
    'في الصف اعتبارًا من',
  'In the room':
    'في القاعة',
  'Include children this number may':
    'الأطفال الذين يجوز لهذا الرقم',
  'Inclusive.':
    'شامل الطرفين.',
  'Insights':
    'قراءات',
  'It leaves the pickers for':
    'يترك المنتقيات الخاصة بـ',
  'It returns to the pickers for':
    'يعيد المنتقيات الخاصة بـ',
  'Its marks are final. The upload will be checked as usual and the service will say what it refuses.':
    'درجات هذا الفصل نهائية. سيُفحص الرفع كالمعتاد وستوضّح الخدمة سبب رفضها.',
  'Labels only. The service takes no capacity here and no rung — moving a class to another rung would carry every enrolment and every mark with it, under a class the children were never in, so it is a new class plus a roster change rather than an edit.':
    'تسميات فقط. لا تأخذ الخدمة سعةً هنا ولا صفًا دراسيًا — نقل صف إلى صف دراسي آخر يحمل معه كل قيد وكل درجة إلى صف لم يكن الأطفال فيه قط، فهو صف جديد مع تعديل كشف، لا تعديلًا.',
  'Last day':
    'آخر يوم',
  'Late':
    'متأخر',
  'Look up':
    'استعلام',
  'Manage':
    'إدارة',
  'Mark the rest present':
    'تسجيل الباقين حاضرين',
  'Marks are recorded against a term, always.':
    'الدرجات تُسجَّل على فصل دراسي، دائمًا.',
  'Marks are recorded against a term, so no mark can be uploaded until one exists.':
    'الدرجات تُسجَّل على فصل دراسي، فلا يمكن رفع أي درجة قبل وجود فصل.',
  "Marks for this term are final. Stated by a person, never derived from the end date — a school enters last term's marks in the first week of this one.":
    'درجات هذا الفصل نهائية. يقرّرها شخص، ولا تُشتق من تاريخ الانتهاء أبدًا — فالمدرسة تُدخل درجات الفصل الماضي في الأسبوع الأول من هذا الفصل.',
  'Move':
    'نقل',
  'Name (Arabic)':
    'الاسم (بالعربية)',
  'Name (English)':
    'الاسم (بالإنجليزية)',
  'New school':
    'مدرسة جديدة',
  'Next':
    'التالي',
  'No academic year exists yet':
    'لا يوجد عام دراسي بعد',
  'No academic years in this school':
    'لا توجد أعوام دراسية في هذه المدرسة',
  'No adult is linked to her':
    'لا يوجد بالغ مرتبط بها',
  'No attendance rate, no subject average, no ranking and no trend. Each of those is a figure the school never stated, computed over a denominator this screen would have chosen for it.':
    'لا نسبة حضور، ولا متوسط لمادة، ولا ترتيب، ولا اتجاه. كل واحد من هذه رقم لم تصرّح به المدرسة، يُحسب على مقام تختاره هذه الشاشة بالنيابة عنها.',
  'No batch open':
    'لا توجد دفعة مفتوحة',
  'No child looked up':
    'لم يُستعلم عن أي طالب',
  'No children for this number':
    'لا يوجد أطفال لهذا الرقم',
  'No class chosen':
    'لم يُختر صف',
  'No class selected':
    'لم يُختر صف',
  'No guardians on file':
    'لا يوجد أولياء أمور مسجّلون',
  'No marks in this window':
    'لا توجد درجات في هذه المدة',
  'No number looked up':
    'لم يُستعلم عن أي رقم',
  'No rows match this filter':
    'لا توجد سطور مطابقة لهذه التصفية',
  'No rung chosen':
    'لم يُختر صف دراسي',
  'No subject rows for this term':
    'لا توجد سطور مواد لهذا الفصل',
  'No subjects in this year':
    'لا توجد مواد في هذا العام',
  'No such academic year':
    'لا يوجد عام دراسي بهذا الرمز',
  'No terms yet':
    'لا توجد فصول دراسية بعد',
  'Nobody here can be told she was absent. Link a guardian from the Guardians screen.':
    'لا أحد هنا يمكن إبلاغه بغيابها. اربط ولي أمر من شاشة أولياء الأمور.',
  'Nobody is in this class yet':
    'لا يوجد أحد في هذا الصف بعد',
  'Nobody is on this register':
    'لا يوجد أحد في هذا الكشف',
  'Nobody is placed in this class':
    'لا يوجد أحد ملحق بهذا الصف',
  'Nobody is recorded for this child. Upload a guardians sheet above to link one.':
    'لا يوجد أحد مسجَّل لهذا الطفل. ارفع ملف أولياء أمور بالأعلى لربط واحد.',
  'Nobody took a register for her between these two dates. Widen the window, or take one from her class.':
    'لم يأخذ أحد كشف حضور لها بين هذين التاريخين. وسّع المدة، أو خذ كشفًا من صفّها.',
  'Nothing can be created until one does — classes, terms and every upload are recorded against a year.':
    'لا يمكن إنشاء شيء قبل وجوده — الصفوف والفصول الدراسية وكل رفع تُسجَّل على عام.',
  'Nothing exists yet. A school comes first: every year, rung, class and mark below it belongs to one.':
    'لا شيء موجود بعد. المدرسة تأتي أولًا: كل عام وصف دراسي وصف ودرجة تحتها تخصّ مدرسة.',
  'Nothing is on file for':
    'لا يوجد شيء مسجَّل لـ',
  'Nothing is written by a preview. Every row is checked and reported first.':
    'المعاينة لا تكتب شيئًا. كل سطر يُفحص ويُبلَّغ عنه أولًا.',
  'Nothing to show':
    'لا يوجد ما يُعرض',
  'Nothing was written. The report below is still readable, but the batch can no longer be committed — upload the file again to get a fresh preview.':
    'لم يُكتب شيء. التقرير أدناه ما زال قابلًا للقراءة، لكن الدفعة لم تعد قابلة للاعتماد — ارفع الملف من جديد للحصول على معاينة جديدة.',
  'On the register':
    'في الكشف',
  "One child's marks":
    'درجات طالب واحد',
  'Open':
    'فتح',
  'Open one from a rung.':
    'افتح واحدًا من صف دراسي.',
  'Open one from a school.':
    'افتح واحدًا من مدرسة.',
  'Open the class to add a child, or upload a roster.':
    'افتح الصف لإضافة طالب، أو ارفع كشف قيد.',
  'Open the full batch in Batches':
    'افتح الدفعة كاملة في شاشة الدفعات',
  'Optional. Empty is not the same as 0.':
    'اختياري. الفراغ ليس صفرًا.',
  'Optional. Empty means the first day of the academic year.':
    'اختياري. الفراغ يعني أول يوم في العام الدراسي.',
  'Optional. Her age is read from this, never stored beside it.':
    'اختياري. يُقرأ عمرها منه، ولا يُخزَّن إلى جانبه أبدًا.',
  'Optional. Leave empty if the sheet has a subject_code column.':
    'اختياري. اتركه فارغًا إذا كان الملف يحوي عمود subject_code.',
  'Optional. Leave empty if the sheet has its own class_code column.':
    'اختياري. اتركه فارغًا إذا كان الملف يحوي عمود class_code خاصًا به.',
  'Optional. Naming one lets the file leave the subject column out.':
    'اختياري. تسمية واحدة تُغني الملف عن عمود المادة.',
  'Optional. Narrows which children the sheet may name.':
    'اختياري. يضيّق نطاق الأطفال الذين يجوز للملف تسميتهم.',
  'Order within the division':
    'الترتيب داخل المرحلة',
  'Pages':
    'الصفحات',
  'Paste a batch id above, or open one from a preview on the Roster, Guardians or Marks screen.':
    'الصق رقم دفعة بالأعلى، أو افتح واحدة من معاينة في شاشة كشف القيد أو أولياء الأمور أو الدرجات.',
  'Phone number':
    'رقم الهاتف',
  'Pick a class to see who is on its register today.':
    'اختر صفًا لعرض من في كشفه اليوم.',
  'Pick one above.':
    'اختر واحدًا بالأعلى.',
  'Placements':
    'الإلحاقات',
  'Placements start':
    'بداية الإلحاق',
  'Placing…':
    'جارٍ الإلحاق…',
  'Present':
    'حاضر',
  'Previous':
    'السابق',
  'Reading the file…':
    'جارٍ قراءة الملف…',
  'Recent on this machine':
    'الأحدث على هذا الجهاز',
  'Record':
    'السجل',
  'Recorded days':
    'الأيام المسجَّلة',
  'Recorded on the link either way.':
    'مسجَّل على الرابط في الاتجاهين.',
  'Refresh':
    'تحديث',
  'Reload':
    'إعادة التحميل',
  'Remove':
    'إزالة',
  'Remove this adult from this child':
    'إزالة هذا البالغ من هذا الطفل',
  'Report-card order':
    'ترتيب شهادة الدرجات',
  'Required — e.g. medical appointment':
    'مطلوب — مثل موعد طبي',
  'Results':
    'النتائج',
  'Revoke':
    'سحب الصلاحية',
  'Rows matching':
    'السطور المطابقة',
  'Rung':
    'صف دراسي',
  'Rung code':
    'رمز الصف الدراسي',
  'Rungs':
    'الصفوف الدراسية',
  'Saving…':
    'جارٍ الحفظ…',
  'School code':
    'رمز المدرسة',
  'Search':
    'بحث',
  'Section suffixes':
    'لواحق الشُعَب',
  'Sections per level':
    'عدد الشُعَب لكل صف دراسي',
  'Sequence':
    'التسلسل',
  'She has never been placed in a class':
    'لم تُلحق بأي صف قط',
  'She is not on any current register. Her record, marks and attendance are unchanged — inactive is a statement about now, not a deletion.':
    'ليست في أي كشف حالي. سجلّها ودرجاتها وحضورها دون تغيير — «غير نشط» تصريح عن الحاضر، لا حذف.',
  'Shown after every preview and every commit.':
    'يظهر بعد كل معاينة وكل اعتماد.',
  'Stated figures, reported exactly as the school recorded them. Nothing here is averaged, weighted or ranked — and a blank is never a zero.':
    'أرقام مُصرَّح بها، تُعرض كما سجّلتها المدرسة تمامًا. لا شيء هنا يُحسب متوسطه أو يُرجَّح أو يُرتَّب — والخانة الفارغة ليست صفرًا أبدًا.',
  'Status':
    'الحالة',
  'Student number':
    'رقم الطالب',
  'Student number or name':
    'رقم الطالب أو الاسم',
  'Subject':
    'المادة',
  'Term':
    'الفصل الدراسي',
  'Terms belong to an academic year, and so does a report card. Pick one in the header.':
    'الفصول الدراسية تخصّ عامًا دراسيًا، وكذلك شهادة الدرجات. اختر عامًا من الشريط العلوي.',
  'Terms sort by this, never by code.':
    'تُرتَّب الفصول الدراسية بهذا، لا بالرمز.',
  "That refusal is the database's, not this screen's — which is why it holds even when two registrars click at once.":
    'هذا الرفض من قاعدة البيانات لا من هذه الشاشة — ولهذا يصمد حتى لو نقر موظفان في اللحظة نفسها.',
  'The adults on file for each child, and which of them may be told what she scored.':
    'البالغون المسجَّلون لكل طفل، ومن منهم يجوز إبلاغه بدرجاته.',
  'The class is already fixed, so the file needs two columns:':
    'الصف محدَّد سلفًا، فالملف يحتاج عمودين:',
  'The code':
    'الرمز',
  'The day her membership starts, not the day you typed it.':
    'اليوم الذي تبدأ فيه عضويتها، لا اليوم الذي كتبته فيه.',
  'The ladder':
    'السلّم',
  'The register':
    'الكشف',
  "The school's own identifier. It never changes.":
    'مُعرِّف المدرسة الخاص. ولا يتغيّر أبدًا.',
  'This is her record, so the change applies in every year rather than only this one. Her marks, her attendance and her placements are untouched.':
    'هذا سجلّها، فالتغيير يسري على كل الأعوام لا على هذا العام وحده. ودرجاتها وحضورها وإلحاقاتها دون مساس.',
  'This preview has expired':
    'انتهت صلاحية هذه المعاينة',
  'This school':
    'هذه المدرسة',
  'This school has no rungs yet':
    'لا توجد صفوف دراسية في هذه المدرسة بعد',
  'This term is closed':
    'هذا الفصل الدراسي مغلق',
  'To class':
    'إلى الصف',
  'Try again':
    'حاول مرة أخرى',
  'Try the student number. A name typed in one script does not match a record that only carries the other.':
    'جرّب رقم الطالب. فالاسم المكتوب بكتابة واحدة لا يطابق سجلًّا لا يحمل إلا الأخرى.',
  'Type a student number and choose a term.':
    'اكتب رقم طالب واختر فصلًا دراسيًا.',
  'Type a student number to see the adults on file for her, and which of them may read her records.':
    'اكتب رقم طالب لعرض البالغين المسجَّلين لها، ومن منهم يجوز له قراءة سجلّاتها.',
  'Type the number that is calling.':
    'اكتب الرقم المتّصل.',
  'Unique across the service, so prefix it when another branch uses the plain code.':
    'فريد على مستوى الخدمة كلها، فأضف إليه بادئة إن كان فرع آخر يستخدم الرمز المجرّد.',
  'Unique within this school; other branches may use the same one.':
    'فريد داخل هذه المدرسة؛ وقد تستخدمه فروع أخرى.',
  'Upload a roster above, choosing this class, to enrol children into it.':
    'ارفع كشف قيد بالأعلى، واختر هذا الصف، لقيد أطفال فيه.',
  'Uploaded':
    'مرفوع',
  'What an upload did, row by row. Filter by outcome to read the rejected rows of a large file without fetching the rest.':
    'ما فعله الرفع، سطرًا سطرًا. صفِّ حسب النتيجة لقراءة السطور المرفوضة من ملف كبير دون جلب الباقي.',
  'Where you are':
    'أين أنت',
  'Which children one number may ask about':
    'الأطفال الذين يجوز لرقم واحد السؤال عنهم',
  'Who may ask about one child':
    'من يجوز له السؤال عن طفل واحد',
  'Work in this year':
    'العمل في هذا العام',
  'Writing…':
    'جارٍ الكتابة…',
  'Written. A second commit of this batch is refused, which is what makes a double-clicked button safe.':
    'كُتبت. ويُرفض اعتماد هذه الدفعة مرة ثانية، وهو ما يجعل النقر المزدوج آمنًا.',
  'Y1 … Yn, the rungs of the school.':
    'Y1 — Yn، صفوف المدرسة الدراسية.',
  'Year code':
    'رمز العام',
  "Year codes are unique across the whole service, so give this school's years a code of their own —":
    'رموز الأعوام فريدة على مستوى الخدمة كلها، فامنح أعوام هذه المدرسة رمزًا خاصًا بها —',
  'Year levels':
    'الصفوف الدراسية',
  'Years':
    'الأعوام',
  'active':
    'نشط',
  'all rows':
    'كل السطور',
  'already there':
    'موجود سلفًا',
  'and a marks upload naming it will be refused.':
    'ورفع درجات يستدعيها سيُرفض.',
  'and is counted as':
    'ويُحسب أنه',
  'and marks uploads will accept it again.':
    'وستقبله عمليات رفع الدرجات من جديد.',
  'and pick one that exists.':
    'واختر واحدًا موجودًا.',
  'closed':
    'مغلق',
  'created':
    'أُنشئ',
  'current':
    'الحالي',
  'division.':
    'مرحلة.',
  'does not change, and cannot: every mark, placement and attendance row in the school points at it. This changes only what the class is called on screen and on a report card.':
    'لا يتغيّر، ولا يمكن أن يتغيّر: كل درجة وإلحاق وسطر حضور في المدرسة يشير إليه. وهذا لا يغيّر إلا اسم الصف على الشاشة وعلى شهادة الدرجات.',
  'first call':
    'أول اتصال',
  'inactive':
    'غير نشط',
  'instead if the relationship is real and only the access is wrong.':
    'بدلًا من ذلك إن كانت الصلة حقيقية والخطأ في الصلاحية وحدها.',
  'instead — it closes this placement and opens the next one together, so she is never in no class at all.':
    'بدلًا من ذلك — فهو يغلق هذا الإلحاق ويفتح التالي معًا، فلا تبقى بلا صف لحظة واحدة.',
  'may read':
    'يجوز له القراءة',
  'may read records':
    'يجوز له قراءة السجلّات',
  'no':
    'لا',
  'no placement covered this term':
    'لم يغطِّ هذا الفصل أي إلحاق',
  'not':
    'ليس',
  'not marked':
    'غير مسجَّل',
  'not yet marked':
    'لم يُسجَّل بعد',
  'open':
    'مفتوح',
  'primary':
    'ابتدائي',
  'read the records of.':
    'قراءة سجلّاتهم.',
  'rejected':
    'مرفوض',
  'restricted':
    'ممنوع',
  'retired':
    'متقاعد',
  'row(s) read':
    'سطر/سطور مقروءة',
  'section(s) in':
    'شُعبة/شُعب في',
  'selected':
    'مختار',
  'term closed':
    'الفصل مغلق',
  'unsaved':
    'غير محفوظ',
  'would each have to be removed first, and the service will refuse while any of them exist.':
    'يجب إزالة كلٍّ منها أولًا، وسترفض الخدمة ما دام أي منها موجودًا.',
  'yes':
    'نعم',
  '— choose a class —':
    '— اختر صفًا —',
  '— choose a term —':
    '— اختر فصلًا دراسيًا —',
  '— from the file —':
    '— من الملف —',

  'Academic years on file':
    'أعوام دراسية مسجَّلة',
  'Adult':
    'البالغ',
  'Age':
    'العمر',
  'Arabic':
    'العربية',
  'Day':
    'اليوم',
  'English':
    'الإنجليزية',
  'In class':
    'في الصف',
  'In the class since':
    'في الصف منذ',
  'In {0}':
    'في {0}',
  'Kind':
    'النوع',
  'Line':
    'السطر',
  'Mark':
    'الدرجة',
  'May read her records':
    'يجوز له قراءة سجلّاتها',
  'Name':
    'الاسم',
  'Outcome':
    'النتيجة',
  'Phone':
    'الهاتف',
  'Pick a year':
    'اختر عامًا',
  'Points':
    'الدرجات',
  'Reason':
    'السبب',
  'Records':
    'السجلّات',
  'Records access':
    'صلاحية الاطّلاع',
  'Relationship':
    'صلة القرابة',
  'Runs':
    'يمتد',
  'Since':
    'منذ',
  'State':
    'الحالة',
  'Stated':
    'مُصرَّح به',
  'Student no.':
    'رقم الطالب',
  'To':
    'إلى',
  'Until':
    'حتى',
  'Why':
    'السبب',
  '{0} class(es)':
    '{0} صف',
  '{0} division(s) in use':
    '{0} مرحلة قيد الاستخدام',
  '{0} in this school':
    '{0} في هذه المدرسة',
  '{0} rung(s)':
    '{0} صف دراسي',

  'Add a child':
    'إضافة طالب',
  'An empty class is a real state, not a missing one — a class exists from the day the ladder is generated and children arrive later. Add one here, or upload the whole roster from the Roster screen.':
    'الصف الفارغ حالة حقيقية لا حالة ناقصة — فالصف موجود من يوم توليد السلّم، والأطفال يصلون بعد ذلك. أضف واحدًا هنا، أو ارفع كشف القيد كاملًا من شاشة كشف القيد.',
  'Classes are generated into a year and every upload names one, so a year comes before anything else.':
    'الصفوف تُولَّد داخل عام، وكل رفع يسمّي عامًا، فالعام يسبق كل شيء.',
  'Edit':
    'تعديل',
  'Generate':
    'توليد',
  'Its academic years, and its ladder grouped by division. Open a rung to see its classes.':
    'أعوامها الدراسية، وسلّمها مجمَّعًا حسب المرحلة. افتح صفًا دراسيًا لعرض صفوفه.',
  'No age can be stated without one.':
    'لا يمكن التصريح بعمر دونه.',
  'No name is on file for her, which is a gap in the record rather than a rendering fault.':
    'لا يوجد اسم مسجَّل لها، وهذا نقص في السجلّ لا خلل في العرض.',
  'No open placement — she is on no current register.':
    'لا يوجد إلحاق مفتوح — فهي ليست في أي كشف حالي.',
  'Open in Batches':
    'فتح في شاشة الدفعات',
  'Open the class':
    'فتح الصف',
  'Place an existing child':
    'إلحاق طالب مسجَّل',
  'Preview':
    'معاينة',
  'Read a class back after uploading':
    'استعراض صف بعد الرفع',
  'Register':
    'الكشف',
  'Table':
    'جدول',
  'Tabs':
    'تبويبات',
  'Template':
    'قالب',
  'This school is not on file.':
    'هذه المدرسة غير مسجَّلة.',
  'This year is not on file.':
    'هذا العام غير مسجَّل.',
  'Upload marks':
    'رفع الدرجات',

  '2. Committed':
    '٢. المعتمد',
  '2. Preview':
    '٢. المعاينة',
  'Add class':
    'إضافة صف',
  'Close':
    'إغلاق',
  'Court order dated…':
    'حكم قضائي بتاريخ…',
  'Edit her record':
    'تعديل سجلّها',
  'Grant':
    'منح',
  'Grant access':
    'منح صلاحية الاطّلاع',
  'Order lifted…':
    'رُفع الحكم…',
  'Rename class':
    'إعادة تسمية الصف',
  'Revoke access':
    'سحب صلاحية الاطّلاع',
  'Why is access being restored?':
    'ما سبب إعادة صلاحية الاطّلاع؟',
  'Why is access being revoked?':
    'ما سبب سحب صلاحية الاطّلاع؟',
  'written':
    'مكتوب',
  'ready':
    'جاهز',
  'Garden':
    'رياض الأطفال',
  'Primary':
    'ابتدائي',
  'Preparatory':
    'إعدادي',
  'Secondary':
    'ثانوي',
  'Not yet grouped':
    'غير مصنَّف بعد',
};
