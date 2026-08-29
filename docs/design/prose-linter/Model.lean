import Std

/-!
An abstract formatter model for the prose-tooling design.

This file is a design-time proof artifact, not a cognitive-miniworld dependency
or CI requirement. It was checked with Lean 4.24.0 using only the bundled Std
library:

    lean docs/design/prose-linter/Model.lean

The operational Markdown scanner and renderer still require Python conformance
and property tests. These theorems state the laws that implementation must
refine; they do not verify Python bytecode.
-/

namespace ProseModel

inductive ProtectedKind where
  | inlineCode
  | codeFence
  | linkDestination
  | html
  | math
deriving DecidableEq, Repr

inductive Segment where
  | prose (tokens : List String) (gaps : List Nat)
  | opaque (kind : ProtectedKind) (raw : String)
deriving DecidableEq, Repr

abbrev Document := List Segment

/- Canonicalization changes only prose-gap widths. -/
def formatSegment : Segment → Segment
  | .prose tokens gaps => .prose tokens (gaps.map fun _ => 1)
  | .opaque kind raw => .opaque kind raw

def format (document : Document) : Document :=
  document.map formatSegment

/- Projection used to state that ordered prose tokens and opaque bytes survive. -/
def contentProjection : Document → List (Sum (List String) String)
  | [] => []
  | .prose tokens _ :: rest => .inl tokens :: contentProjection rest
  | .opaque _ raw :: rest => .inr raw :: contentProjection rest

/- Stronger projection that also preserves each protected region's kind. -/
def protectedProjection : Document → List (ProtectedKind × String)
  | [] => []
  | .prose _ _ :: rest => protectedProjection rest
  | .opaque kind raw :: rest => (kind, raw) :: protectedProjection rest

def gapsCanonical : Segment → Bool
  | .prose _ gaps => gaps.all (· == 1)
  | .opaque _ _ => true

def formatterClean : Document → Bool
  | [] => true
  | segment :: rest => gapsCanonical segment && formatterClean rest

theorem formatSegment_idempotent (segment : Segment) :
    formatSegment (formatSegment segment) = formatSegment segment := by
  cases segment <;> simp [formatSegment]

theorem format_idempotent (document : Document) :
    format (format document) = format document := by
  simp [format, formatSegment_idempotent]

theorem content_preserved (document : Document) :
    contentProjection (format document) = contentProjection document := by
  induction document with
  | nil => rfl
  | cons head tail ih =>
      rw [show format (head :: tail) = formatSegment head :: format tail by rfl]
      cases head <;> simp [formatSegment, contentProjection, ih]

theorem protected_preserved (document : Document) :
    protectedProjection (format document) = protectedProjection document := by
  induction document with
  | nil => rfl
  | cons head tail ih =>
      rw [show format (head :: tail) = formatSegment head :: format tail by rfl]
      cases head <;> simp [formatSegment, protectedProjection, ih]

theorem format_is_clean (document : Document) :
    formatterClean (format document) = true := by
  induction document with
  | nil => rfl
  | cons head tail ih =>
      rw [show format (head :: tail) = formatSegment head :: format tail by rfl]
      cases head <;>
        simp [formatSegment, formatterClean, gapsCanonical, ih]

end ProseModel
