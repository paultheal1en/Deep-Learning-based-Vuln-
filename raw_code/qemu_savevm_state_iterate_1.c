int qemu_savevm_state_iterate(QEMUFile *f)

{

    SaveStateEntry *se;

    int ret = 1;



    trace_savevm_state_iterate();

    QTAILQ_FOREACH(se, &savevm_state.handlers, entry) {

        if (!se->ops || !se->ops->save_live_iterate) {

            continue;

        }

        if (se->ops && se->ops->is_active) {

            if (!se->ops->is_active(se->opaque)) {

                continue;

            }

        }

        if (qemu_file_rate_limit(f)) {

            return 0;

        }

        trace_savevm_section_start(se->idstr, se->section_id);



        save_section_header(f, se, QEMU_VM_SECTION_PART);



        ret = se->ops->save_live_iterate(f, se->opaque);

        trace_savevm_section_end(se->idstr, se->section_id, ret);




        if (ret < 0) {

            qemu_file_set_error(f, ret);

        }

        if (ret <= 0) {

            /* Do not proceed to the next vmstate before this one reported

               completion of the current stage. This serializes the migration

               and reduces the probability that a faster changing state is

               synchronized over and over again. */

            break;

        }

    }

    return ret;

}