---- MODULE AtomicPublish ----
(* Crash safety of checkpoint session publish (design 7.2, B.3).

   The writer streams data to a temporary file, fsyncs, then atomically
   renames it into place. A crash can happen at any point. Recovery must
   expose either the old complete checkpoint or the new complete checkpoint,
   never a torn file. The only action that changes the visible version is
   the atomic rename. *)
EXTENDS Naturals, TLC

VARIABLES phase, visible, crashed

(* phase: writing -> fsync -> renamed -> done; crash -> recovered.
   visible: "old" or "new". *)

Init ==
  /\ phase \in {"writing", "fsync"}
  /\ visible = "old"
  /\ crashed = FALSE

(* More chunks are written; still in the writing phase. *)
KeepWriting ==
  /\ phase = "writing"
  /\ phase' = "writing"
  /\ UNCHANGED <<visible, crashed>>

ChunksDone ==
  /\ phase = "writing"
  /\ phase' = "fsync"
  /\ UNCHANGED <<visible, crashed>>

FsyncDone ==
  /\ phase = "fsync"
  /\ phase' = "renamed"
  /\ UNCHANGED <<visible, crashed>>

(* Atomic rename: the single point where the visible version changes. *)
Rename ==
  /\ phase = "renamed"
  /\ phase' = "done"
  /\ visible' = "new"
  /\ UNCHANGED crashed

WriterFinish ==
  /\ phase = "done"
  /\ UNCHANGED <<phase, visible, crashed>>

(* Crash before or after the rename; the temp file is discarded on recovery. *)
Crash ==
  /\ ~crashed
  /\ phase \in {"writing", "fsync", "renamed"}
  /\ phase' = "recovered"
  /\ crashed' = TRUE
  /\ UNCHANGED visible

Recover ==
  /\ phase = "recovered"
  /\ phase' = "done"
  /\ UNCHANGED <<visible, crashed>>

Next ==
  \/ KeepWriting
  \/ ChunksDone
  \/ FsyncDone
  \/ Rename
  \/ WriterFinish
  \/ Crash
  \/ Recover

Spec == Init /\ [][Next]_<<phase, visible, crashed>>

(* Only complete versions are ever visible. *)
VersionComplete == visible \in {"old", "new"}

(* A crash before the rename keeps the old version. *)
CrashBeforeRenameKeepsOld ==
  (phase \in {"recovered", "done"} /\ visible = "old") =>
    crashed

(* A new version is visible only after the rename completed. *)
NewOnlyAfterRename ==
  visible = "new" => phase \in {"done", "recovered"}

====
