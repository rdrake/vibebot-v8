---------------------------- MODULE PendingTasks ----------------------------
(* pending_tasks lease protocol: persistence.py claim/release/update and
   service.py _run_pending_poll + plugin.py _deliver_pending_result.

   Workers: each is one poll pass (sweep -> phase 1 -> phase 2 -> deliver).
   `_pending_poll_lock` serializes passes of ONE plugin instance. A second
   worker models the zombie poll left behind by die() (2s drain) during an
   @reload: a different instance, a different lock, the same database.

   Every row write is `UPDATE ... WHERE id = ?`. With OwnerCheck = TRUE they
   become `WHERE id = ? AND claimed_until = <my lease>` (the proposed fix).  *)
EXTENDS Naturals, FiniteSets

CONSTANTS Tasks, Workers, Lease, Expiry, MaxT, OwnerCheck

VARIABLES now, st, cu, na, exp, payload,     \* the pending_tasks row
          wpc, pnow, myLease, c1, c2, res,   \* per-worker poll state
          sentOK, sentFail, expiredMsg,      \* what the user saw
          provCalls,                         \* completed (billed) provider calls
          staleWrites                        \* row writes by a worker whose lease was taken over

vars == <<now, st, cu, na, exp, payload, wpc, pnow, myLease, c1, c2, res,
          sentOK, sentFail, expiredMsg, provCalls, staleWrites>>

Live == {"pending", "ready", "retrying"}

Init ==
  /\ now = 0
  /\ st = [t \in Tasks |-> "pending"] /\ cu = [t \in Tasks |-> 0]
  /\ na = [t \in Tasks |-> 0]         /\ exp = [t \in Tasks |-> Expiry]
  /\ payload = [t \in Tasks |-> "none"]
  /\ wpc = [w \in Workers |-> "idle"] /\ pnow = [w \in Workers |-> 0]
  /\ myLease = [w \in Workers |-> 0]
  /\ c1 = [w \in Workers |-> {}] /\ c2 = [w \in Workers |-> {}]
  /\ res = [w \in Workers |-> [t \in Tasks |-> "none"]]
  /\ sentOK = [t \in Tasks |-> 0] /\ sentFail = [t \in Tasks |-> 0]
  /\ expiredMsg = [t \in Tasks |-> 0] /\ provCalls = [t \in Tasks |-> 0]
  /\ staleWrites = 0

Tick == now < MaxT /\ now' = now + 1
        /\ UNCHANGED <<st, cu, na, exp, payload, wpc, pnow, myLease, c1, c2, res,
                       sentOK, sentFail, expiredMsg, provCalls, staleWrites>>

\* Owns(w,t): the conditional-UPDATE guard. Without the fix every write lands.
Owns(w, t) == ~OwnerCheck \/ cu[t] = myLease[w]

\* delete_expired_pending_tasks + claim_due_pending_tasks(pending), one step
\* per BEGIN IMMEDIATE is a simplification: both see the same `now`.
Start(w) ==
  /\ wpc[w] = "idle"
  /\ LET gone == {t \in Tasks : st[t] \in Live /\ exp[t] <= now}
         due  == {t \in Tasks : t \notin gone /\ st[t] = "pending"
                                /\ na[t] <= now /\ cu[t] <= now}
     IN /\ st' = [t \in Tasks |-> IF t \in gone THEN "gone" ELSE st[t]]
        /\ expiredMsg' = [t \in Tasks |-> expiredMsg[t] + IF t \in gone THEN 1 ELSE 0]
        /\ cu' = [t \in Tasks |-> IF t \in due THEN now + Lease ELSE cu[t]]
        /\ c1' = [c1 EXCEPT ![w] = due]
  /\ pnow' = [pnow EXCEPT ![w] = now]
  /\ myLease' = [myLease EXCEPT ![w] = now + Lease]
  /\ wpc' = [wpc EXCEPT ![w] = "p1"]
  /\ UNCHANGED <<now, na, exp, payload, c2, res, sentOK, sentFail, provCalls, staleWrites>>

\* One provider call for a claimed row; outcome chosen by the provider.
P1(w) ==
  /\ wpc[w] = "p1"
  /\ \E t \in c1[w], outcome \in {"completed", "failed_terminal", "transient"} :
       /\ provCalls' = [provCalls EXCEPT ![t] = @ + IF outcome = "completed" THEN 1 ELSE 0]
       /\ c1' = [c1 EXCEPT ![w] = @ \ {t}]
       /\ staleWrites' = staleWrites + IF st[t] # "gone" /\ Owns(w, t) /\ cu[t] # myLease[w] THEN 1 ELSE 0
       /\ IF st[t] = "gone" \/ ~Owns(w, t)
            THEN UNCHANGED <<st, cu, na, payload>>
          ELSE IF outcome = "transient"
            THEN /\ na' = [na EXCEPT ![t] = now + 1] /\ cu' = [cu EXCEPT ![t] = 0]
                 /\ UNCHANGED <<st, payload>>
            ELSE /\ st' = [st EXCEPT ![t] = "ready"] /\ cu' = [cu EXCEPT ![t] = 0]
                 /\ payload' = [payload EXCEPT ![t] = outcome]
                 /\ UNCHANGED na
  /\ UNCHANGED <<now, exp, wpc, pnow, myLease, c2, res, sentOK, sentFail, expiredMsg>>

\* Phase 2 claim uses the top-of-pass `now` (pnow), exactly as the code does.
P2Claim(w) ==
  /\ wpc[w] = "p1" /\ c1[w] = {}
  /\ LET due == {t \in Tasks : st[t] \in {"ready", "retrying"}
                               /\ na[t] <= pnow[w] /\ cu[t] <= pnow[w]}
     IN /\ cu' = [t \in Tasks |-> IF t \in due THEN pnow[w] + Lease ELSE cu[t]]
        /\ c2' = [c2 EXCEPT ![w] = due]
        /\ res' = [res EXCEPT ![w] = [t \in Tasks |-> IF t \in due THEN payload[t] ELSE "none"]]
  /\ myLease' = [myLease EXCEPT ![w] = pnow[w] + Lease]
  /\ wpc' = [wpc EXCEPT ![w] = "p2"]
  /\ UNCHANGED <<now, st, na, exp, payload, pnow, c1, sentOK, sentFail, expiredMsg, provCalls, staleWrites>>

\* _deliver_pending_result: send, then ack (delete). A zombie worker's sends
\* are dropped by _safe_queue (closing=True) and it then writes nothing.
Deliver(w) ==
  /\ wpc[w] = "p2"
  /\ \E t \in c2[w] :
       /\ c2' = [c2 EXCEPT ![w] = @ \ {t}]
       /\ IF w = "zombie"
            THEN UNCHANGED <<st, sentOK, sentFail>>
            ELSE /\ IF res[w][t] = "completed"
                      THEN sentOK' = [sentOK EXCEPT ![t] = @ + 1] /\ UNCHANGED sentFail
                      ELSE sentFail' = [sentFail EXCEPT ![t] = @ + 1] /\ UNCHANGED sentOK
                 /\ st' = [st EXCEPT ![t] = "gone"]   \* the ack stays unguarded: a zombie never delivers
  /\ UNCHANGED <<now, cu, na, exp, payload, wpc, pnow, myLease, c1, res, expiredMsg, provCalls, staleWrites>>

Finish(w) ==
  /\ wpc[w] = "p2" /\ c2[w] = {}
  /\ wpc' = [wpc EXCEPT ![w] = "idle"]
  /\ UNCHANGED <<now, st, cu, na, exp, payload, pnow, myLease, c1, c2, res,
                 sentOK, sentFail, expiredMsg, provCalls, staleWrites>>

\* The zombie only ever runs the one pass it was in when die() returned.
CanStart(w) == IF w = "zombie" THEN myLease[w] = 0 /\ myLease["main"] = 0 ELSE ("zombie" \in Workers => myLease["zombie"] # 0)

Next ==
  \/ Tick
  \/ \E w \in Workers :
       \/ (CanStart(w) /\ Start(w))
       \/ P1(w) \/ P2Claim(w) \/ Deliver(w) \/ Finish(w)

Spec == Init /\ [][Next]_vars

\* ---------------------------------------------------------------------------
\* The user gets exactly one answer: never both a success and a failure line,
\* never an answer plus an "expired" notice, never the same answer twice.
OneAnswer == \A t \in Tasks : sentOK[t] + sentFail[t] + expiredMsg[t] <= 1

\* A result the provider completed (and billed) is never answered with a failure.
NoSuccessOverwritten == \A t \in Tasks : ~(provCalls[t] >= 1 /\ sentFail[t] >= 1)

\* At most one billed, completed provider call per task.
NoDuplicateBilling == \A t \in Tasks : provCalls[t] <= 1

\* A provider result is only written by the worker holding the row's lease.
NoStaleWrite == staleWrites = 0
=============================================================================
