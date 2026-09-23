/-
  Model of plugin.py `_finish_irc_line` (pastebin branch) with
  `_trim_long_reply_teaser` + service.py `truncate_to_word_boundary`.

  Python `len()` counts code points, so `List Char` + `List.length` is exact.
  The IRC budget `allowed` (supybot.reply.mores.length) is enforced by
  Limnoria's `ircutils.wrap`, which counts UTF-8 BYTES.

  Claim under test: the link line fits one wire line, i.e.
      bytes line ≤ allowed.
-/
namespace IrcLine

def bytes : List Char → Nat
  | [] => 0
  | c :: cs => c.utf8Size + bytes cs

theorem bytes_append (a b : List Char) : bytes (a ++ b) = bytes a + bytes b := by
  induction a with
  | nil => simp [bytes]
  | cons c cs ih => simp [bytes, ih]; omega

theorem bytes_reverse (s : List Char) : bytes s.reverse = bytes s := by
  induction s with
  | nil => rfl
  | cons c cs ih => simp [bytes_append, ih, bytes]; omega

theorem bytes_dropWhile_le (p : Char → Bool) (s : List Char) :
    bytes (s.dropWhile p) ≤ bytes s := by
  induction s with
  | nil => simp [bytes]
  | cons c cs ih =>
    simp only [List.dropWhile]
    split
    · simp [bytes]; omega
    · exact Nat.le_refl _

/-- Python `s.rstrip(chars)`. -/
def rstripBy (p : Char → Bool) (s : List Char) : List Char :=
  (s.reverse.dropWhile p).reverse

theorem bytes_rstripBy_le (p : Char → Bool) (s : List Char) :
    bytes (rstripBy p s) ≤ bytes s := by
  unfold rstripBy
  rw [bytes_reverse]
  have := bytes_dropWhile_le p s.reverse
  rw [bytes_reverse] at this
  exact this

/-- `idx = s.rfind(" "); s[:idx] if idx > 0 else s` -/
def cutAtLastSpace (s : List Char) : List Char :=
  match s.reverse.dropWhile (· != ' ') with
  | [] => s
  | _ :: rest => if rest = [] then s else rest.reverse

theorem bytes_cutAtLastSpace_le (s : List Char) : bytes (cutAtLastSpace s) ≤ bytes s := by
  unfold cutAtLastSpace
  have h := bytes_dropWhile_le (· != ' ') s.reverse
  rw [bytes_reverse] at h
  split
  · exact Nat.le_refl _
  · rename_i c rest heq
    split
    · exact Nat.le_refl _
    · rw [heq] at h
      rw [bytes_reverse]
      simp [bytes] at h
      omega

def label : List Char := "Full answer".toList
def isSpace (c : Char) : Bool := c == ' '
def isTrail (c : Char) : Bool := c == ' ' || c == ',' || c == ';' || c == ':' || c == '-'

/-! ## The pre-fix code (character budgets) -/

/-- service.py `truncate_to_word_boundary(text, max_chars)` -/
def truncWord (text : List Char) (maxC : Nat) : List Char :=
  if maxC = 0 ∨ text.length ≤ maxC then text
  else rstripBy isSpace (cutAtLastSpace (rstripBy isSpace (text.take maxC)))

/-- plugin.py `_trim_long_reply_teaser` (input already whitespace-collapsed). -/
def trimTeaser (teaser : List Char) (maxC : Nat) : List Char :=
  let teaser := if teaser = [] then label else teaser
  let tr := truncWord teaser maxC
  let tr := if tr ≠ teaser then rstripBy isTrail tr else tr
  if tr = [] then label else tr

/-- plugin.py `_finish_irc_line`, pastebin branch, save succeeded (`url ≠ ""`). -/
def finishCurrent (allowed : Nat) (nick teaser url : List Char) (cap : Option Nat) : List Char :=
  let suffix := " - Full answer: ".toList ++ url
  let maxC := allowed - suffix.length - nick.length
  let maxC := match cap with | some c => min c maxC | none => maxC
  if maxC = 0 then nick ++ "Full answer: ".toList ++ url
  else nick ++ trimTeaser teaser maxC ++ suffix

/-! ### Counterexamples (the kernel evaluates them; no `native_decide`) -/

-- 1. Multibyte teaser: every char of "日本語" is 3 bytes. Budget 40.
def cjk : List Char := "日本語の説明日本語の説明日本語の説明".toList
#eval (bytes (finishCurrent 40 "a: ".toList cjk "u".toList none),
       String.ofList (finishCurrent 40 "a: ".toList cjk "u".toList none))
example : ¬ bytes (finishCurrent 40 "a: ".toList cjk "u".toList none) ≤ 40 := by decide

-- 2. ASCII-only: a teaser that trims to nothing falls back to "Full answer"
--    (11 chars) regardless of the budget (here 2 chars of room).
def dashes : List Char := "--- x".toList
example : trimTeaser dashes 2 = label := by decide
example : ¬ bytes (finishCurrent 22 "a: ".toList dashes "u".toList none) ≤ 22 := by decide

/-! ## The fix (byte budgets) and its proof -/

/-- Longest prefix of `s` whose UTF-8 encoding fits in `b` bytes. -/
def takeBytes : Nat → List Char → List Char
  | _, [] => []
  | b, c :: cs => if c.utf8Size ≤ b then c :: takeBytes (b - c.utf8Size) cs else []

theorem bytes_takeBytes_le (b : Nat) (s : List Char) : bytes (takeBytes b s) ≤ b := by
  induction s generalizing b with
  | nil => simp [takeBytes, bytes]
  | cons c cs ih =>
    simp only [takeBytes]
    split
    · have := ih (b - c.utf8Size); simp [bytes]; omega
    · simp [bytes]

def truncWordB (text : List Char) (maxB : Nat) : List Char :=
  if bytes text ≤ maxB then text
  else rstripBy isSpace (cutAtLastSpace (rstripBy isSpace (takeBytes maxB text)))

theorem bytes_truncWordB_le (text : List Char) (maxB : Nat) :
    bytes (truncWordB text maxB) ≤ maxB := by
  unfold truncWordB
  split
  · assumption
  · have h1 := bytes_rstripBy_le isSpace (takeBytes maxB text)
    have h2 := bytes_cutAtLastSpace_le (rstripBy isSpace (takeBytes maxB text))
    have h3 := bytes_rstripBy_le isSpace (cutAtLastSpace (rstripBy isSpace (takeBytes maxB text)))
    have h4 := bytes_takeBytes_le maxB text
    omega

theorem bytes_trail_le (src tr : List Char) (maxB : Nat) (h : bytes tr ≤ maxB) :
    bytes (if tr ≠ src then rstripBy isTrail tr else tr) ≤ maxB := by
  split
  · exact Nat.le_trans (bytes_rstripBy_le _ _) h
  · exact h

/-- plugin.py: `teaser = truncate_to_byte_budget(raw, max_bytes)`, then
    `.rstrip(" ,;:-")` if that cut anything. -/
def clip (raw : List Char) (maxB : Nat) : List Char :=
  let tr := truncWordB raw maxB
  if tr ≠ raw then rstripBy isTrail tr else tr

theorem bytes_clip_le (raw : List Char) (maxB : Nat) : bytes (clip raw maxB) ≤ maxB :=
  bytes_trail_le _ _ _ (bytes_truncWordB_le _ _)

/-- plugin.py `_finish_irc_line` as fixed, pastebin branch, save succeeded.
    `teaserFn` is arbitrary: in production it is an LLM summary. -/
def finishFixed (allowed : Nat) (nick content url : List Char) (cap : Option Nat)
    (teaserFn : List Char → Nat → List Char) : List Char :=
  let suffix := " - Full answer: ".toList ++ url
  let labelLine := nick ++ "Full answer: ".toList ++ url
  let maxB := allowed - bytes suffix - bytes nick
  let maxC := match cap with | some c => min c maxB | none => maxB
  if maxC = 0 then labelLine
  else
    let t := clip (teaserFn content maxC) maxB
    if t = [] then labelLine else nick ++ t ++ suffix

/-- **Theorem.** Whenever the nick prefix and the link suffix fit the budget at
    all, the fixed line fits one wire line — for every teaser function,
    content, URL, and cap. -/
theorem finishFixed_fits (allowed : Nat) (nick content url : List Char) (cap : Option Nat)
    (teaserFn : List Char → Nat → List Char)
    (h : bytes nick + bytes (" - Full answer: ".toList ++ url) ≤ allowed) :
    bytes (finishFixed allowed nick content url cap teaserFn) ≤ allowed := by
  have hs : bytes (" - Full answer: ".toList ++ url)
      = 3 + bytes ("Full answer: ".toList ++ url) := by
    have e : " - Full answer: ".toList ++ url = ' ' :: '-' :: ' ' :: ("Full answer: ".toList ++ url) := rfl
    have h1 : (' ').utf8Size = 1 := by decide
    have h2 : ('-').utf8Size = 1 := by decide
    rw [e]; simp only [bytes]; omega
  have hlabel : bytes (nick ++ "Full answer: ".toList ++ url) ≤ allowed := by
    simp only [bytes_append] at hs h ⊢; omega
  have key : ∀ m, bytes (if m = 0 then nick ++ "Full answer: ".toList ++ url
      else if clip (teaserFn content m) (allowed - bytes (" - Full answer: ".toList ++ url) - bytes nick) = []
        then nick ++ "Full answer: ".toList ++ url
        else nick ++ clip (teaserFn content m) (allowed - bytes (" - Full answer: ".toList ++ url) - bytes nick)
          ++ (" - Full answer: ".toList ++ url)) ≤ allowed := by
    intro m
    split
    · exact hlabel
    · split
      · exact hlabel
      · have := bytes_clip_le (teaserFn content m)
          (allowed - bytes (" - Full answer: ".toList ++ url) - bytes nick)
        simp only [bytes_append] at this h ⊢
        omega
  unfold finishFixed
  cases cap with
  | none => exact key _
  | some c => exact key _

end IrcLine
