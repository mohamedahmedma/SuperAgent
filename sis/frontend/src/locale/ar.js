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
  /* -- Signing in, and what a person's roles reach ---------------------------------
   *
   * صلاحية is the word an Egyptian school uses for a permission, and دور for a role. The
   * refusal sentences name who to ask rather than what went wrong: a teacher reading
   * "you do not have permission" has nothing to do next, and the person who can fix it is
   * always whoever assigns roles at their own school. */
  'Sign in': 'تسجيل الدخول',
  'Sign in failed': 'تعذر تسجيل الدخول',
  'Show password': 'إظهار كلمة المرور',
  'Hide password': 'إخفاء كلمة المرور',
  'Signing in…': 'جارٍ تسجيل الدخول…',
  'Sign out': 'تسجيل الخروج',
  'Signing in shows you the classes and screens your roles cover.':
    'تسجيل الدخول يعرض لك الفصول والشاشات التي تغطيها أدوارك.',
  'No role': 'بلا دور',
  'Not your screen': 'هذه الشاشة ليست ضمن صلاحياتك',
  'Your roles do not cover this part of the console. Ask whoever manages roles at your school if you need it.':
    'أدوارك لا تشمل هذا الجزء من النظام. راجع المسؤول عن الأدوار في مدرستك إن كنت تحتاج إليه.',
  'You can read this register but not record it. Ask whoever manages roles at your school for the classes you take.':
    'يمكنك الاطلاع على هذا الكشف دون تسجيله. راجع المسؤول عن الأدوار في مدرستك لإضافة الفصول التي تدرّسها.',
  /* The six roles, as a school names them. */
  'System Administrator': 'مدير النظام',
  'School Owner': 'مالك المدرسة',
  'School Manager': 'مدير المدرسة',
  'Grade Supervisor': 'مشرف الصف',
  'Class Supervisor': 'مشرف الصف',
  'A class supervisor may also remain an ordinary teacher.': 'يمكن أن يكون مشرف الصف معلّمًا أيضًا.',
  'Attendance Supervisor': 'مشرف الحضور والغياب',
  'Choose a grade': 'اختر السنة الدراسية',
  'This account will only see this grade’s classes and students in attendance.':
    'سيظهر لهذا الحساب فصول وطلاب هذه السنة فقط في الغياب.',
  Teacher: 'معلّم',
  'Subject Coordinator': 'منسّق المادة',
  Username: 'اسم المستخدم',
  Password: 'كلمة المرور',
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
  'Settings — appearance and language': 'الإعدادات — المظهر واللغة',
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
  Language: 'اللغة',
  'Choose a clear light or dark appearance. Your choice is remembered in this browser.':
    'اختر المظهر الفاتح أو الداكن. سيُحفظ اختيارك في هذا المتصفح.',
  'Choose the interface language and reading direction.':
    'اختر لغة الواجهة واتجاه القراءة.',
  'Latin names first': 'الأسماء اللاتينية أولًا',
  'Arabic names first, right to left': 'الأسماء العربية أولًا، من اليمين إلى اليسار',
  'Remembered in this browser. Nothing here is sent to the service.':
    'تُحفظ هذه الإعدادات في هذا المتصفح، ولا يُرسل منها شيء إلى الخدمة.',
  Done: 'تم',

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
  'Arabic and Languages':
    'عربي ولغات',
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
  'Choose one student and guardian roster sheet':
    'اختر ملفًا واحدًا للطلاب وأولياء الأمور',
  'One row per student. Include the guardian in the same row; alternate guardian fields remain optional.':
    'صف واحد لكل طالب، ويتضمن ولي الأمر في الصف نفسه، وتظل بيانات الاتصال البديلة اختيارية.',
  'student_number,full_name_ar,full_name_en,class_code,guardian_name_ar,guardian_name_en,guardian_phone,relationship_type,is_primary_contact,can_view_records':
    'student_number,full_name_ar,full_name_en,class_code,guardian_name_ar,guardian_name_en,guardian_phone,relationship_type,is_primary_contact,can_view_records',
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
  'Educational levels':
    'المراحل التعليمية',
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
  'Friday':
    'الجمعة',
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
  'Languages':
    'لغات',
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
  'Monday':
    'الاثنين',
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
  'Number of academic terms':
    'عدد الفصول الدراسية',
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
  'Saturday':
    'السبت',
  'Saving…':
    'جارٍ الحفظ…',
  'School code':
    'رمز المدرسة',
  'School language type':
    'نظام اللغة بالمدرسة',
  'School working days':
    'أيام العمل بالمدرسة',
  'Search':
    'بحث',
  'Section suffixes':
    'لواحق الشُعَب',
  'Sections per level':
    'عدد الشُعَب لكل صف دراسي',
  'Select at least one educational level.':
    'اختر مرحلة تعليمية واحدة على الأقل.',
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
  'Sunday':
    'الأحد',
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
  'Thursday':
    'الخميس',
  'To class':
    'إلى الصف',
  'Try again':
    'حاول مرة أخرى',
  'Try the student number. A name typed in one script does not match a record that only carries the other.':
    'جرّب رقم الطالب. فالاسم المكتوب بكتابة واحدة لا يطابق سجلًّا لا يحمل إلا الأخرى.',
  'Tuesday':
    'الثلاثاء',
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
  'Wednesday':
    'الأربعاء',
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
  '{0} grade(s)':
    '{0} صف/صفوف',
  '{0} in this school':
    '{0} في هذه المدرسة',
  '{0} term(s)':
    '{0} فصل/فصول',
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
  'That did not work':
    'لم تنجح العملية',
  'Loading…':
    'جارٍ التحميل…',
  'Choose a term':
    'اختر فصلًا دراسيًا',
  'Choose a class':
    'اختر صفًا',
  'Nothing changed':
    'لم يتغير شيء',
  'Guardian removed':
    'تمت إزالة ولي الأمر',
  '{0} found':
    'تم العثور على {0}',
  'Open {0}':
    'فتح {0}',
  'Open {0} {1}':
    'فتح {0} {1}',
  'Open class {0}':
    'فتح الصف {0}',
  'Open academic year {0}':
    'فتح العام الدراسي {0}',
  'Nothing matches “{0}”':
    'لا توجد نتائج تطابق «{0}»',
  '{0} in the record — newest first':
    '{0} في السجل — الأحدث أولًا',
  '{0} on her contact list':
    '{0} في قائمة التواصل',
  'Counts of what the school recorded — nothing derived':
    'أعداد ما سجلته المدرسة — دون بيانات مستنتجة',
  'Edit {0}':
    'تعديل {0}',
  'Move {0}':
    'نقل {0}',
  'Rename {0}':
    'إعادة تسمية {0}',
  'New class on {0}':
    'صف جديد في {0}',
  'No classes on {0} in {1}':
    'لا توجد صفوف في {0} خلال {1}',
  '{0} on this rung':
    '{0} في هذا الصف الدراسي',
  'Classes on this rung in {0}. A rung belongs to the school and outlives every year; a class belongs to the year.':
    'صفوف هذا المستوى في {0}. المستوى تابع للمدرسة ويستمر بين الأعوام، أما الصف فتابع للعام الدراسي.',
  'School {0} saved':
    'تم حفظ المدرسة {0}',
  'Academic year {0} saved':
    'تم حفظ العام الدراسي {0}',
  'Rung {0} saved':
    'تم حفظ المستوى {0}',
  '{0} moved to {1}':
    'تم نقل {0} إلى {1}',
  'Could not move {0}':
    'تعذّر نقل {0}',
  'New academic year in {0}':
    'عام دراسي جديد في {0}',
  'New rung in {0}':
    'مستوى دراسي جديد في {0}',
  'Term {0} saved':
    'تم حفظ الفصل الدراسي {0}',
  'Subject {0} added to {1}':
    'تمت إضافة المادة {0} إلى {1}',
  'Terms — {0}':
    'الفصول الدراسية — {0}',
  'Subjects taught in {0}':
    'المواد التي تُدرّس في {0}',
  '{0} in this year':
    '{0} في هذا العام',
  'Batch {0}':
    'الدفعة {0}',
  '{0} placed in {1}':
    'تم إلحاق {0} بالصف {1}',
  '{0} renamed':
    'تمت إعادة تسمية {0}',
  '{0} removed from {1}':
    'تمت إزالة {0} من {1}',
  'Nobody is in {0} yet':
    'لا يوجد أحد في {0} بعد',
  'Marks for {0}':
    'درجات {0}',
  '{0} is closed':
    '{0} مغلق',
  'Everything this class does, in {0}.':
    'كل ما يخص هذا الصف في {0}.',
  '{0} updated':
    'تم تحديث {0}',
  '{0} guardian(s) on file':
    '{0} من أولياء الأمور مسجلون',
  '{0} child(ren)':
    '{0} من الطلاب',
  'Garden':
    'KG',
  'Primary':
    'ابتدائي',
  'Preparatory':
    'إعدادي',
  'Secondary':
    'ثانوي',
  'Not yet grouped':
    'غير مصنَّف بعد',
  'Academic track':
    'المسار الأكاديمي',
  'You are managing this structure independently.':
    'أنت تدير هذا الهيكل بشكل مستقل.',
  'Assign subjects to grades': 'تعيين المواد للصفوف',
  'Available subjects': 'المواد المتاحة',
  'Remove assignment': 'إزالة التعيين',
  'Drop subjects here': 'اسحب المواد هنا',

  /* The subject board. */
  'Which grades teach what':
    'الصفوف وما تدرّسه',
  'A subject is taught only where it is placed':
    'تُدرَّس المادة حيث تُعيَّن فقط',
  'A subject appears only where it is assigned. Physics assigned to Secondary does not appear in Primary, and the two academic tracks are assigned separately.':
    'تظهر المادة حيث تُعيَّن فقط. الفيزياء المعيَّنة للثانوي لا تظهر في الابتدائي، ويُعيَّن المساران الأكاديميان كلٌّ على حدة.',
  'Drag a subject onto a grade, or tap it to pick it up.':
    'اسحب المادة إلى الصف، أو انقر عليها لاختيارها.',
  'Now choose a grade below, or tap the subject again to put it back.':
    'اختر صفًا بالأسفل، أو انقر المادة مرة أخرى لإعادتها.',
  'No active subject in this year to assign.':
    'لا توجد مادة مفعَّلة في هذا العام لتعيينها.',
  'Teaches nothing yet.':
    'لا يُدرَّس فيه شيء بعد.',
  'Assign here':
    'عيِّن هنا',
  'Already here':
    'معيَّنة هنا بالفعل',
  'Remove {0} from {1}':
    'إزالة {0} من {1}',
  'No grades on this school yet':
    'لا توجد صفوف في هذه المدرسة بعد',
  'A subject is assigned to a grade, so the ladder has to exist first. Add rungs on the school screen, or generate them below.':
    'تُعيَّن المادة إلى صف، فلا بد من وجود الصفوف أولًا. أضِف الصفوف من شاشة المدرسة، أو أنشئها بالأسفل.',
  'No grades in this track':
    'لا توجد صفوف في هذا المسار',
  'Add a rung to this track on the school screen, and it will appear here.':
    'أضِف صفًا إلى هذا المسار من شاشة المدرسة ليظهر هنا.',
  'Optional. Only what {0} is assigned to teach.':
    'اختياري. فقط ما عُيِّن لتدريسه في {0}.',

  /* Stage 6 — the year's terms, and what the year is attached to. */
  'Optional':
    'اختياري',
  'Term {0}':
    'الفصل {0}',
  'Save dates':
    'حفظ التواريخ',
  'No dates yet. The term still holds marks and still closes — dates are only needed to say when it runs.':
    'لا توجد تواريخ بعد. الفصل يستقبل الدرجات ويُغلق كالمعتاد — التواريخ لازمة فقط لبيان موعده.',
  'These sections come from the number of terms the school runs. Dates are optional — a term works without them, and they can be filled in whenever the calendar is settled.':
    'تُنشأ هذه الأقسام من عدد الفصول التي تعمل بها المدرسة. التواريخ اختيارية — يعمل الفصل بدونها، ويمكن إدخالها متى استقر التقويم.',
  '{0} term section(s), one panel each':
    '{0} قسم فصلي، لكلٍّ لوحته',
  'none yet':
    'لا شيء بعد',
  'Add a term':
    'إضافة فصل',
  'Terms are created with the year, from the number the school runs. If none are here, this year predates that — add one, or re-save the year from the school screen.':
    'تُنشأ الفصول مع العام الدراسي من العدد الذي تعمل به المدرسة. إن لم يكن هنا شيء فهذا العام أقدم من ذلك — أضِف فصلًا، أو أعِد حفظ العام من شاشة المدرسة.',
  'This year is attached to':
    'يرتبط هذا العام بـ',
  'School, tracks, grades and classes — read together':
    'المدرسة والمسارات والصفوف والفصول — تُقرأ معًا',
  '{0} selected by the school':
    '{0} اختارتها المدرسة',
  'Terms':
    'الفصول',
  'Grades':
    'الصفوف',
  'Tracks':
    'المسارات',
  'Not yet in a track':
    'خارج المسارات بعد',
  '{0} grade(s), {1} class(es)':
    '{0} صف، {1} فصل',
  'No grades on this track yet.':
    'لا توجد صفوف في هذا المسار بعد.',
  'Teacher roles': 'أدوار المعلّمين',
  'Staff roles': 'أدوار العاملين',
  'Choose a grade and subject to review its teachers. Supervisors are managed separately and may also be teachers.':
    'اختر الصف والمادة لعرض معلّميها. وتُدار قائمة المشرفين بشكل مستقل، ويمكن أن يكون المشرف معلّماً أيضاً.',
  'Teachers': 'المعلّمون',
  'Teachers come from their subject and class assignments.': 'تُحدد قائمة المعلّمين حسب المواد والفصول المسندة إليهم.',
  'Teacher subject': 'مادة المعلّم',
  'All subjects': 'كل المواد',
  'No teachers match this grade and subject.': 'لا يوجد معلّمون مطابقون لهذا الصف وهذه المادة.',
  'Supervisors': 'المشرفون',
  'A supervisor may also remain an ordinary teacher. Roles are additive.': 'يمكن أن يكون المشرف معلّماً عادياً أيضاً؛ فالأدوار تُضاف ولا يستبدل أحدها الآخر.',
  'Supervisor scope': 'نطاق الإشراف',
  'Roles are additive. Selecting a supervisor role keeps the Teacher role active.':
    'الأدوار تراكمية؛ اختيار دور إشرافي يُبقي دور المعلّم فعالًا.',
  'Active roles': 'الأدوار الفعالة',
  'Active': 'فعال',
  'No login account': 'لا يوجد حساب دخول',
  'Create or link a login account before assigning roles.':
    'أنشئ حساب دخول أو اربطه قبل تعيين الأدوار.',
  'Teacher attendance': 'حضور المعلّمين',
  '{0} record(s)': '{0} سجل',
  'Date': 'التاريخ',
  'Note': 'ملاحظة',
  'Class assignments': 'تعيينات الفصول',
  'Choose a managed grade, subject, eligible teacher, and one or more classes.':
    'اختر صفًا دراسيًا تحت إشرافك، ثم المادة والمعلّم المؤهل وفصلًا أو أكثر.',
  'No managed grades are assigned to this account.':
    'لم تُعيَّن لهذا الحساب صفوف دراسية للإشراف عليها.',
  '1. Grade': '١. الصف الدراسي',
  '2. Subject': '٢. المادة',
  '3. Eligible teacher': '٣. المعلّم المؤهل',
  '4. Classes': '٤. الفصول',
  'Assign teacher': 'تعيين المعلّم',
  'Teaching staff on this grade': 'هيئة التدريس في هذا الصف',
  'Read-only. Each teacher is shown as they stand on this grade alone.':
    'للاطّلاع فقط. يظهر كل معلّم بما يخصّ هذا الصف وحده.',
  'Teacher': 'المعلّم',
  'Staff number': 'الرقم الوظيفي',
  'Subjects': 'المواد',
  'Assigned classes': 'الفصول المسندة',

  /* Stage 13 — taking the register. */
  'Take attendance': 'رصد الحضور',
  'Choose a day, a grade and a class. Mark the children who are here; the rest are recorded absent.':
    'اختر اليوم والصف الدراسي والفصل. علّم الطلاب الحاضرين، ويُسجَّل الباقون غائبين.',
  '1. Day, grade and class': '١. اليوم والصف والفصل',
  'Grade': 'الصف الدراسي',
  'done': 'مكتمل',
  'No classes are assigned to this account.': 'لا توجد فصول مسندة لهذا الحساب.',
  'A register is taken by whoever holds the class. Ask whoever manages roles at your school for the classes you take.':
    'يرصد الحضورَ من يُسند إليه الفصل. راجع مسؤول الصلاحيات في مدرستك بشأن الفصول التي ترصدها.',
  'This register was already taken for this day. Saving again corrects it rather than recording it twice.':
    'سبق رصد الحضور في هذا اليوم. الحفظ مرة أخرى يصحّح الرصد ولا يكرّره.',
  'Finish — rest absent': 'إنهاء — الباقون غائبون',
  'Records every child still blank as absent.': 'يسجّل كل طالب لم يُعلَّم بعد بوصفه غائبًا.',
  '{0} change(s) not yet saved.': '{0} تغييرات لم تُحفظ بعد.',
  'Nothing is written until you save. Saving again corrects the day rather than adding a second set of marks.':
    'لن يُكتب شيء قبل الحفظ. والحفظ مرة أخرى يصحّح اليوم ولا يضيف سجلاً مكرراً.',
  '{0} present': '{0} حاضر',
  '{0} absent': '{0} غائب',
  '{0} late': '{0} متأخر',
  '{0} excused': '{0} بعذر',
  '{0} not yet marked': '{0} لم يُرصد بعد',
  'An unmarked child shows {0}. The counts use the {1} recorded marks, not all {2} children.':
    'يظهر الطالب غير المرصود بالعلامة {0}. وتعتمد الأعداد على {1} سجلات محفوظة، لا على كل الطلاب وعددهم {2}.',
  'Either this class is empty, or nobody was placed in it on {0}.':
    'إما أن الفصل فارغ، أو لم يكن أي طالب مقيداً فيه يوم {0}.',
  'name not on file': 'الاسم غير مسجل',
  /* Stage 14 — teacher academic access and mark entry. */
  'Only your assigned classes and subjects are shown.': 'تظهر فقط الفصول والمواد المسندة إليك.',
  'Enter class marks': 'إدخال درجات الفصل',
  'Assigned class and subject': 'الفصل والمادة المسندان',
  'No teaching assignments': 'لا توجد تكليفات تدريسية',
  'Ask your school manager to assign your subjects and classes.': 'اطلب من مدير المدرسة إسناد موادك وفصولك.',
  'Student': 'الطالب',
  'Save marks': 'حفظ الدرجات',
  'Marks saved.': 'تم حفظ الدرجات.',
  'Upload this class marks file': 'رفع ملف درجات هذا الفصل',
  'CSV only. Columns: student_number, percentage. Only the selected class and your assigned subject are accepted.':
    'ملف CSV فقط بالأعمدة student_number وpercentage. لا تُقبل إلا درجات الفصل المحدد والمادة المسندة إليك.',
  'The file has no student rows.': 'الملف لا يحتوي على صفوف طلاب.',
  'Required columns: student_number and percentage.': 'الأعمدة المطلوبة: student_number وpercentage.',
  'Every percentage must be between 0 and 100.': 'يجب أن تكون كل درجة بين 0 و100.',
  'Could not read the marks file.': 'تعذرت قراءة ملف الدرجات.',
  '{0} marks loaded from the file.': 'تم تحميل {0} درجات من الملف.',
  /* Stage 15 — principal teacher eligibility workflow. */
  'Teacher setup': 'إعداد المعلّم',
  'Define the teacher account, subjects, eligible grades, and track scope. Grade supervisors assign classes afterward.':
    'حدّد حساب المعلّم ومواده والصفوف المؤهل لها ونطاق المسار، ثم يعيّن موجّه الصف الفصول لاحقًا.',
  'Teacher account': 'حساب المعلّم',
  'Existing teacher': 'معلّم موجود',
  'Create a teacher': 'إنشاء معلّم',
  'English name': 'الاسم بالإنجليزية',
  'Arabic name': 'الاسم بالعربية',
  'Required for a new account; leave blank to keep an existing password.':
    'مطلوبة للحساب الجديد؛ اتركها فارغة للاحتفاظ بكلمة المرور الحالية.',
  'Subject, grade, and track eligibility': 'أهلية المادة والصف والمسار',
  'Choose a compatible subject and grade': 'اختر مادة وصفًا متوافقين',
  'Add eligibility': 'إضافة أهلية',
  'No eligible grades yet': 'لا توجد صفوف مؤهلة بعد',
  'Track': 'المسار',
  'Save teacher configuration': 'حفظ إعداد المعلّم',
  'Teacher configuration saved.': 'تم حفظ إعداد المعلّم.',
  'Assessment name': 'اسم التقييم',
  'Choose grade': 'اختر الصف',
  'Choose stage': 'اختر المرحلة',
  'Choose subject': 'اختر المادة',
  'Create teacher': 'إنشاء معلّم',
  'Every mark must be between zero and the maximum mark.': 'يجب أن تكون كل درجة بين صفر والحد الأقصى للدرجات.',
  'Find a child in your classes': 'ابحث عن طالب في فصولك',
  'January monthly exam': 'الاختبار الشهري لشهر يناير',
  'Maximum mark': 'الحد الأقصى للدرجات',
  'New teacher account': 'حساب معلّم جديد',
  'This staff number already belongs to another teacher.': 'هذا الرقم الوظيفي مسجّل بالفعل لمعلّم آخر.',
  'Timetable': 'الجدول',
  'Choose a class to view its weekly timetable. Supervisors can drag subjects into lessons and swap existing lessons.': 'اختر الفصل لعرض جدوله الأسبوعي. يستطيع المشرف سحب المواد إلى الحصص وتبديل الحصص الموجودة.',
  'Class and term': 'الفصل والفصل الدراسي',
  'Term': 'الفصل الدراسي',
  'Editable timetable': 'جدول قابل للتعديل',
  'View only': 'عرض فقط',
  'Teachers only see classes assigned to them. Supervisors see classes in their managed grade.': 'يرى المدرس الفصول المسندة إليه فقط، ويرى المشرف فصول الصف الذي يشرف عليه.',
  'Drag a subject onto any lesson. Drag one lesson onto another to swap them.': 'اسحب المادة إلى أي حصة، أو اسحب حصة إلى أخرى لتبديلهما.',
  'Weekly timetable': 'الجدول الأسبوعي',
  'Period': 'الحصة',
  'Drop subject here': 'ضع المادة هنا',
  'Clear lesson': 'حذف الحصة',
  'Break': 'فسحة',
  'Sunday': 'الأحد', 'Monday': 'الاثنين', 'Tuesday': 'الثلاثاء',
  'Wednesday': 'الأربعاء', 'Thursday': 'الخميس', 'Friday': 'الجمعة', 'Saturday': 'السبت',
};
