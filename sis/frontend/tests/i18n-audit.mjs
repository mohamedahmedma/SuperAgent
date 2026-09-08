import fs from 'node:fs';
import path from 'node:path';
import { AR } from '../src/locale/ar.js';

const root = path.resolve(import.meta.dirname, '../src');
const files = [];

function walk(directory) {
  for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) walk(target);
    else if (/\.(js|jsx)$/.test(entry.name)) files.push(target);
  }
}

walk(root);
const keys = new Set();
for (const file of files) {
  const source = fs.readFileSync(file, 'utf8');
  for (const match of source.matchAll(/\bt\(\s*(['"])(.*?)\1/g)) keys.add(match[2]);
}

const intentionallyLatin = new Set([
  '#',
  'NC-2025-2026',
  'Open|the class',
  'percentage',
  'student_number',
  'student_number,phone,full_name_ar,full_name_en,relationship,is_primary_contact',
  'student_number,subject_code,percentage'
]);
const missing = [...keys]
  .filter((key) => !Object.hasOwn(AR, key) && !intentionallyLatin.has(key))
  .sort();
const invalid = Object.entries(AR)
  .filter(([, value]) => !String(value).trim() || /\?{3,}/.test(value))
  .map(([key]) => key)
  .sort();
const tokens = (value) => [...String(value).matchAll(/\{\d+\}/g)]
  .map((match) => match[0])
  .sort()
  .join(',');
const tokenMismatches = Object.entries(AR)
  .filter(([key, value]) => tokens(key) !== tokens(value))
  .map(([key]) => key)
  .sort();

console.log(
  `Translation calls: ${keys.size}; missing Arabic entries: ${missing.length}; invalid Arabic entries: ${invalid.length}; token mismatches: ${tokenMismatches.length}`
);
for (const key of missing) console.log(key);
for (const key of invalid) console.log(`Invalid translation: ${key}`);
for (const key of tokenMismatches) console.log(`Token mismatch: ${key}`);

if (missing.length || invalid.length || tokenMismatches.length) process.exitCode = 1;
