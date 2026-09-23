---- MODULE MCAdmissionFALSE ----
EXTENDS AnimateAdmission
MCOwner == [r \in {"a1","a2","b1"} |-> IF r = "b1" THEN "bob" ELSE "alice"]
====
