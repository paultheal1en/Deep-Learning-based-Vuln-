void qemu_savevm_state_cancel(Monitor *mon, QEMUFile *f)

{

    SaveStateEntry *se;



    QTAILQ_FOREACH(se, &savevm_handlers, entry) {

        if (se->save_live_state) {

            se->save_live_state(mon, f, -1, se->opaque);

        }

    }

}
