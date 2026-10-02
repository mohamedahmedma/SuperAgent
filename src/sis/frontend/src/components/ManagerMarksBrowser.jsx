import { useEffect, useState } from 'react';
import { api } from '../api.js';
import { Store } from '../store.js';
import { labelOf, pickName, useQuery, useResource, useStore } from '../hooks.js';
import { Card, Empty, ErrorNote, Field, Select, Table } from './Ui.jsx';
import { t } from '../i18n.js';

export function ManagerMarksBrowser({ fixedClassCode = '', fixedYearLevel = '' }) {
  const state = useStore(); const year = state.year;
  const [grade,setGrade]=useState(fixedYearLevel||''); const [classCode,setClassCode]=useState(fixedClassCode||'');
  const [term,setTerm]=useState(''); const [subject,setSubject]=useState('');
  const [assessmentType,setAssessmentType]=useState(''); const [assessmentId,setAssessmentId]=useState('');

  // Grade supervisors are bounded to the YEAR_LEVEL scope(s) carried by their
  // signed-in profile. Do not broaden the request to the whole academic year.
  // Principal/owner/admin behaviour remains unchanged even if roles are additive.
  const heldRoles=Store.roles();
  const profileRoles=(state.profile&&Array.isArray(state.profile.roles))?state.profile.roles:[];
  const isYearSupervisor=
    heldRoles.indexOf('floor_supervisor')>=0 &&
    heldRoles.indexOf('school_manager')<0 &&
    heldRoles.indexOf('school_owner')<0 &&
    heldRoles.indexOf('admin')<0;

  const supervisorGrades=isYearSupervisor
    ? Array.from(new Set(profileRoles
        .filter((row)=>row.role_code==='floor_supervisor'&&row.scope_type==='year_level'&&row.scope_code)
        .map((row)=>String(row.scope_code))))
    : [];

  const supervisorDefaultGrade=supervisorGrades[0]||'';

  const classesKey=isYearSupervisor
    ? `${Store.keys.classes(year)}:year-supervisor:${supervisorGrades.join(',')||'none'}`
    : Store.keys.classes(year);

  const classes=useResource(
    classesKey,
    ()=>isYearSupervisor
      ? Promise.all(supervisorGrades.map((code)=>api.classes(year,code))).then((groups)=>groups.flat())
      : api.classes(year),
    !!(year&&(!isYearSupervisor||supervisorGrades.length))
  );

  const allClasses=classes.value||[];
  const visibleClasses=isYearSupervisor
    ? allClasses.filter((x)=>supervisorGrades.indexOf(String(x.year_level_code))>=0)
    : allClasses;

  const fixedClass=visibleClasses.find((x)=>x.code===fixedClassCode);

  const actualGrade=fixedYearLevel||(
    isYearSupervisor
      ? (grade||supervisorDefaultGrade)
      : ((fixedClass&&fixedClass.year_level_code)||grade)
  );

  useEffect(()=>{
    if(fixedClassCode)setClassCode(fixedClassCode);
    if(fixedYearLevel)setGrade(fixedYearLevel);
    else if(fixedClass&&fixedClass.year_level_code)setGrade(fixedClass.year_level_code);
  },[fixedClassCode,fixedYearLevel,fixedClass&&fixedClass.year_level_code]);

  useEffect(()=>{
    if(
      isYearSupervisor &&
      supervisorDefaultGrade &&
      supervisorGrades.indexOf(String(grade))<0
    ){
      setGrade(supervisorDefaultGrade);
      if(!fixedClassCode)setClassCode('');
    }
  },[isYearSupervisor,supervisorDefaultGrade,supervisorGrades.join(','),grade,fixedClassCode]);

  const grades=Array.from(new Map(visibleClasses.map((x)=>[
    x.year_level_code,
    {
      value:x.year_level_code,
      label:x.year_level_name_ar||x.year_level_name_en||x.year_level_code
    }
  ])).values());

  const classOptions=visibleClasses
    .filter((x)=>!actualGrade||x.year_level_code===actualGrade)
    .map((x)=>({value:x.code,label:labelOf(x,state.lang)||x.code}));
  const terms=useResource(Store.keys.terms(year),()=>api.terms(year),!!year);
  useEffect(()=>{if(!term&&terms.value&&terms.value.length)setTerm(terms.value[0].code);},[term,(terms.value||[]).length]);
  const subjects=useResource(actualGrade?Store.keys.gradeSubjects(year,actualGrade):Store.keys.subjects(year,false),()=>api.subjects(year,false,actualGrade||null),!!year);
  useEffect(()=>{setSubject('');setAssessmentType('');setAssessmentId('');},[classCode,actualGrade]);
  useEffect(()=>setAssessmentId(''),[term,subject,assessmentType]);
  const assessments=useQuery(()=>api.classAssessments(classCode,year,term,subject,assessmentType),[classCode,year,term,subject,assessmentType],!!(classCode&&year&&term&&subject&&assessmentType));
  const assessmentRows=(assessments.value&&assessments.value.assessments)||[];
  const sheet=useQuery(()=>api.classAssessment(classCode,assessmentId,year),[classCode,assessmentId,year],!!(classCode&&assessmentId&&year));
  return <Card title={t('Class marks')} subtitle={t('Read only')} tight>
    <div className="card-body"><div className="row g-3">
      {!fixedClassCode?<><Field className="col-12 col-md-3" label={t('Grade')} required><Select value={grade} strict options={grades} placeholder={t('— choose grade —')} onChange={(v)=>{setGrade(v);setClassCode('');}} /></Field>
      <Field className="col-12 col-md-3" label={t('Class')} required><Select value={classCode} strict options={classOptions} placeholder={t('— choose class —')} onChange={setClassCode} /></Field></>:null}
      <Field className="col-12 col-md-3" label={t('Term')} required><Select value={term} strict options={(terms.value||[]).map((x)=>({value:x.code,label:labelOf(x,state.lang)}))} onChange={setTerm} /></Field>
      <Field className="col-12 col-md-3" label={t('Subject')} required><Select value={subject} strict options={(subjects.value||[]).map((x)=>({value:x.code,label:labelOf(x,state.lang)}))} placeholder={t('— choose subject —')} onChange={setSubject} /></Field>
      <Field className="col-12 col-md-3" label={t('Assessment type')} required><Select value={assessmentType} strict options={[{value:'exam',label:t('Exam')},{value:'assignment',label:t('Homework assignment')}]} placeholder={t('— choose assessment type —')} onChange={setAssessmentType} /></Field>
      <Field className="col-12 col-md-3" label={t('Assessment')} required><Select value={assessmentId} strict options={assessmentRows.map((x)=>({value:String(x.id),label:`${x.name}${x.max_points==null?'':` / ${x.max_points}`}`}))} placeholder={assessmentRows.length?t('— choose assessment —'):t('No assessments recorded')} onChange={setAssessmentId} /></Field>
    </div></div>
    <ErrorNote error={classes.error||terms.error||subjects.error||assessments.error||sheet.error} onRetry={()=>{classes.reload();terms.reload();subjects.reload();assessments.reload();sheet.reload();}} />
    {!classCode||!subject||!assessmentType?<Empty title={t('Choose the filters above')}>{t('Choose grade, class, subject and assessment type to read the recorded marks.')}</Empty>:
    assessmentType&&!assessments.loading&&!assessmentRows.length?<Empty title={t('No assessments recorded')}>{t('No saved assessments of this type exist yet.')}</Empty>:
    <Table loading={sheet.loading} rows={(sheet.value&&sheet.value.students)||[]} rowKey={(x)=>x.student_number} columns={[
      {key:'number',header:t('Student number'),cell:(x)=>x.student_number},{key:'name',header:t('Student'),cell:(x)=>pickName(x,state.lang)},
      {key:'points',header:t('Points'),cell:(x)=>x.is_graded?`${x.points??'—'}${x.max_points==null?'':` / ${x.max_points}`}`:'—'},
      {key:'percentage',header:t('Percentage'),cell:(x)=>x.is_graded&&x.percentage!=null?`${Number(x.percentage).toFixed(1)}%`:'—'}]} />}
  </Card>;
}
