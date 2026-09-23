------------------------- MODULE TypingHoldsSendLock -------------------------
(* Proposed fix: state changes stay under _lock; every send runs under a
   separate _send_lock and reads the CURRENT state (under _lock) right
   before sending. Transitions send the current state; the keepalive sends
   "active" only if the key is still held when it reads.                   *)
EXTENDS Naturals, FiniteSets
CONSTANTS Holders
Threads == Holders \cup {"k"}
VARIABLES count, wire, pc, kpc, sendOwner, pendingVal

vars == <<count, wire, pc, kpc, sendOwner, pendingVal>>

Init == count = 0 /\ wire = "done" /\ pc = [h \in Holders |-> "acq"]
        /\ kpc = "idle" /\ sendOwner = "none" /\ pendingVal = "none"

Acquire(h) == pc[h] = "acq" /\ count' = count + 1
  /\ pc' = [pc EXCEPT ![h] = (IF count = 0 THEN "sendLock" ELSE "work")]
  /\ UNCHANGED <<wire, kpc, sendOwner, pendingVal>>
Work(h) == pc[h] = "work" /\ pc' = [pc EXCEPT ![h] = "rel"]
  /\ UNCHANGED <<count, wire, kpc, sendOwner, pendingVal>>
Release(h) == pc[h] = "rel" /\ count' = count - 1
  /\ pc' = [pc EXCEPT ![h] = (IF count = 1 THEN "sendLockR" ELSE "end")]
  /\ UNCHANGED <<wire, kpc, sendOwner, pendingVal>>

\* take _send_lock, then read the current state under _lock
SLock(h) == pc[h] \in {"sendLock", "sendLockR"} /\ sendOwner = "none"
  /\ sendOwner' = h /\ pendingVal' = (IF count > 0 THEN "active" ELSE "done")
  /\ pc' = [pc EXCEPT ![h] = (IF pc[h] = "sendLock" THEN "emit" ELSE "emitR")]
  /\ UNCHANGED <<count, wire, kpc>>
Emit(h) == pc[h] \in {"emit", "emitR"} /\ wire' = pendingVal
  /\ sendOwner' = "none" /\ pendingVal' = "none"
  /\ pc' = [pc EXCEPT ![h] = (IF pc[h] = "emit" THEN "work" ELSE "end")]
  /\ UNCHANGED <<count, kpc>>

\* keepalive: take _send_lock, read "held?", send active only if held
KLock == kpc = "idle" /\ sendOwner = "none" /\ sendOwner' = "k"
  /\ pendingVal' = (IF count > 0 THEN "active" ELSE "none")
  /\ kpc' = "emit" /\ UNCHANGED <<count, wire, pc>>
KEmit == kpc = "emit" /\ wire' = (IF pendingVal = "none" THEN wire ELSE pendingVal)
  /\ sendOwner' = "none" /\ pendingVal' = "none" /\ kpc' = "idle"
  /\ UNCHANGED <<count, pc>>

Next == \/ \E h \in Holders : Acquire(h) \/ Work(h) \/ Release(h) \/ SLock(h) \/ Emit(h)
        \/ KLock \/ KEmit
Spec == Init /\ [][Next]_vars /\ WF_vars(KLock) /\ WF_vars(KEmit)
        /\ \A h \in Holders : WF_vars(Work(h) \/ Release(h) \/ SLock(h) \/ Emit(h))

Quiescent == (\A h \in Holders : pc[h] = "end") /\ kpc = "idle"
NoStuckActive == Quiescent => wire = "done"
\* stronger than before: no thread mid-send => wire matches the refcount
NoSendInFlight == sendOwner = "none" /\ \A h \in Holders : pc[h] \notin {"sendLock", "sendLockR"}
WireMatches == NoSendInFlight => (wire = "active") = (count > 0)
HeldHeals == [](count > 0 ~> (wire = "active" \/ count = 0))
=============================================================================
