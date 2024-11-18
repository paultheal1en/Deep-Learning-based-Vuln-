int qemu_savevm_state_begin(Monitor *mon, QEMUFile *f, int blk_enable,

                            int shared)

{

    SaveStateEntry *se;

    int ret;



    QTAILQ_FOREACH(se, &savevm_handlers, entry) {

        if(se->set_params == NULL) {

            continue;

	}

	se->set_params(blk_enable, shared, se->opaque);

    }

    

    qemu_put_be32(f, QEMU_VM_FILE_MAGIC);

    qemu_put_be32(f, QEMU_VM_FILE_VERSION);



    QTAILQ_FOREACH(se, &savevm_handlers, entry) {

        int len;



        if (se->save_live_state == NULL)

            continue;



        /* Section type */

        qemu_put_byte(f, QEMU_VM_SECTION_START);

        qemu_put_be32(f, se->section_id);



        /* ID string */

        len = strlen(se->idstr);

        qemu_put_byte(f, len);

        qemu_put_buffer(f, (uint8_t *)se->idstr, len);



        qemu_put_be32(f, se->instance_id);

        qemu_put_be32(f, se->version_id);



        ret = se->save_live_state(mon, f, QEMU_VM_SECTION_START, se->opaque);

        if (ret < 0) {

            qemu_savevm_state_cancel(mon, f);

            return ret;

        }

    }

    ret = qemu_file_get_error(f);

    if (ret != 0) {

        qemu_savevm_state_cancel(mon, f);

    }



    return ret;



}
