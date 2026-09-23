-------------------------- MODULE AnimateAdmission --------------------------
(* plugin.py _animate_for_assistant -> _animate_admission -> service.py
   video_generation -> _stash_timeout -> db.save_pending_task.

   The per-user cap (animateMaxPendingPerUser) and global cap
   (animateMaxPending) are checked with COUNT reads; the row that makes
   the request count is INSERTed only after an HTTP submit to the video box.
   Limnoria runs commands threaded, so two requests can interleave.       *)
EXTENDS Naturals, FiniteSets

CONSTANTS Requests, Owner, PerUserCap, GlobalCap, Atomic

VARIABLES rows, pc

vars == <<rows, pc>>

Mine(u) == Cardinality({r \in rows : Owner[r] = u})

Init == rows = {} /\ pc = [r \in Requests |-> "check"]

\* count_pending_animate_for / count_pending_animate. With Atomic = TRUE the
\* check and the INSERT are one step (the fix: reserve the row first).
Check(r) ==
  /\ pc[r] = "check"
  /\ IF Mine(Owner[r]) < PerUserCap /\ Cardinality(rows) < GlobalCap
       THEN IF Atomic
              THEN rows' = rows \cup {r} /\ pc' = [pc EXCEPT ![r] = "done"]
              ELSE rows' = rows /\ pc' = [pc EXCEPT ![r] = "submit"]
       ELSE rows' = rows /\ pc' = [pc EXCEPT ![r] = "refused"]

\* POST to the box (seconds), then save_pending_task.
Submit(r) ==
  /\ pc[r] = "submit"
  /\ pc' = [pc EXCEPT ![r] = "insert"]
  /\ UNCHANGED rows

Insert(r) ==
  /\ pc[r] = "insert"
  /\ rows' = rows \cup {r}
  /\ pc' = [pc EXCEPT ![r] = "done"]

Next == \E r \in Requests : Check(r) \/ Submit(r) \/ Insert(r)

Spec == Init /\ [][Next]_vars

PerUserCapHolds == \A u \in {Owner[r] : r \in Requests} : Mine(u) <= PerUserCap
GlobalCapHolds  == Cardinality(rows) <= GlobalCap
=============================================================================
