int qemu_savevm_state_iterate(Monitor *mon, QEMUFile *f)

{

    SaveStateEntry *se;

    int ret = 1;



    QTAILQ_FOREACH(se, &savevm_handlers, entry) {

        if (se->save_live_state == NULL)

            continue;



        /* Section type */

        qemu_put_byte(f, QEMU_VM_SECTION_PART);

        qemu_put_be32(f, se->section_id);



        ret = se->save_live_state(mon, f, QEMU_VM_SECTION_PART, se->opaque);

        if (!ret) {

            /* Do not proceed to the next vmstate before this one reported

               completion of the current stage. This serializes the migration

               and reduces the probability that a faster changing state is

               synchronized over and over again. */

            break;

        }

    }

    if (ret != 0) {

        return ret;

    }

    ret = qemu_file_get_error(f);

    if (ret != 0) {

        qemu_savevm_state_cancel(mon, f);

    }

    return ret;

}
