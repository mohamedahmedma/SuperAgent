export type UserRole = 'user' | 'admin' | 'parent' | 'staff';

export interface CurrentUser {
  username: string;
  role: UserRole;
  /**
   * Present only for a parent whose account an administrator has bound to a guardian.
   * Purely informational here — it tells the UI whether to offer the student-records
   * features. Authority comes from the signed claim in the token, which the records
   * facade checks; nothing may be granted on the strength of this field.
   */
  guardianId?: string | null;
  displayName?: string;
}
