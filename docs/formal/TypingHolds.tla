---------------------------- MODULE TypingHolds ----------------------------
(* The PRE-FIX typing_holds.py, for ONE (network,target) key. TLC finds
   NoStuckActive violated; TypingHoldsSendLock.tla is the shipped fix.

   Holders run hold() -> work -> release(). _acquire/_release mutate the
   refcount under _lock and send +typing OUTSIDE the lock. The keepalive
   thread snapshots _ircs under _lock, then sends "active" outside it.

   `wire` is the last +typing state the client received. queueMsg is FIFO,
   so each send step atomically overwrites it.                              *)
EXTENDS Naturals, FiniteSets

CONSTANTS Holders

VARIABLES count, wire, pc, kpc, ksnap

vars == <<count, wire, pc, kpc, ksnap>>

Init ==
  /\ count = 0
  /\ wire  = "done"
  /\ pc    = [h \in Holders |-> "acq"]
  /\ kpc   = "snap"
  /\ ksnap = FALSE

\* _acquire: locked increment; `first` decides whether to send.
Acquire(h) ==
  /\ pc[h] = "acq"
  /\ count' = count + 1
  /\ pc' = [pc EXCEPT ![h] = IF count = 0 THEN "sendActive" ELSE "work"]
  /\ UNCHANGED <<wire, kpc, ksnap>>

SendActive(h) ==
  /\ pc[h] = "sendActive"
  /\ wire' = "active"
  /\ pc' = [pc EXCEPT ![h] = "work"]
  /\ UNCHANGED <<count, kpc, ksnap>>

\* the command's own reply goes out, then stop_typing() -> release()
Work(h) ==
  /\ pc[h] = "work"
  /\ pc' = [pc EXCEPT ![h] = "rel"]
  /\ UNCHANGED <<count, wire, kpc, ksnap>>

\* _release: locked decrement; `last` decides whether to send "done".
Release(h) ==
  /\ pc[h] = "rel"
  /\ count' = count - 1
  /\ pc' = [pc EXCEPT ![h] = IF count = 1 THEN "sendDone" ELSE "end"]
  /\ UNCHANGED <<wire, kpc, ksnap>>

SendDone(h) ==
  /\ pc[h] = "sendDone"
  /\ wire' = "done"
  /\ pc' = [pc EXCEPT ![h] = "end"]
  /\ UNCHANGED <<count, kpc, ksnap>>

\* keepalive: `snapshot = list(self._ircs.items())` under the lock ...
KSnap ==
  /\ kpc = "snap"
  /\ ksnap' = (count > 0)
  /\ kpc' = "send"
  /\ UNCHANGED <<count, wire, pc>>

\* ... then `_safe_send(irc, target, "active")` for every snapshotted key.
KSend ==
  /\ kpc = "send"
  /\ wire' = IF ksnap THEN "active" ELSE wire
  /\ kpc' = "snap"
  /\ UNCHANGED <<count, pc, ksnap>>

Next ==
  \/ \E h \in Holders :
       Acquire(h) \/ SendActive(h) \/ Work(h) \/ Release(h) \/ SendDone(h)
  \/ KSnap \/ KSend

Spec == Init /\ [][Next]_vars /\ WF_vars(KSnap) /\ WF_vars(KSend)
        /\ \A h \in Holders : WF_vars(SendActive(h) \/ Work(h) \/ Release(h) \/ SendDone(h))

\* ---------------------------------------------------------------------------
TypeOK == count \in 0..Cardinality(Holders) /\ wire \in {"active", "done"}

\* The refcount equals the number of holders between acquire and release.
RefcountExact ==
  count = Cardinality({h \in Holders : pc[h] \in {"sendActive", "work", "rel"}})

\* SAFETY: once every holder is finished and the keepalive thread is not
\* mid-send, the client must have seen "done". A violation is a typing
\* indicator left on after the bot has replied and let go.
Quiescent == (\A h \in Holders : pc[h] = "end") /\ kpc = "snap"
NoStuckActive == Quiescent => wire = "done"

\* LIVENESS: while someone holds the target the client sees "active" again
\* (the keepalive heals a "done" that raced past a fresh acquire).
HeldHeals == [](count > 0 ~> (wire = "active" \/ count = 0))
=============================================================================
