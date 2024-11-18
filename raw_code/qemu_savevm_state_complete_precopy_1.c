void qemu_savevm_state_complete_precopy(QEMUFile *f, bool iterable_only)

{

    QJSON *vmdesc;

    int vmdesc_len;

    SaveStateEntry *se;

    int ret;

    bool in_postcopy = migration_in_postcopy();



    trace_savevm_state_complete_precopy();



    cpu_synchronize_all_states();



    QTAILQ_FOREACH(se, &savevm_state.handlers, entry) {

        if (!se->ops ||

            (in_postcopy && se->ops->save_live_complete_postcopy) ||

            (in_postcopy && !iterable_only) ||

            !se->ops->save_live_complete_precopy) {

            continue;

        }



        if (se->ops && se->ops->is_active) {

            if (!se->ops->is_active(se->opaque)) {

                continue;

            }

        }

        trace_savevm_section_start(se->idstr, se->section_id);



        save_section_header(f, se, QEMU_VM_SECTION_END);



        ret = se->ops->save_live_complete_precopy(f, se->opaque);

        trace_savevm_section_end(se->idstr, se->section_id, ret);

        save_section_footer(f, se);

        if (ret < 0) {

            qemu_file_set_error(f, ret);

            return;

        }

    }



    if (iterable_only) {

        return;

    }



    vmdesc = qjson_new();

    json_prop_int(vmdesc, "page_size", qemu_target_page_size());

    json_start_array(vmdesc, "devices");

    QTAILQ_FOREACH(se, &savevm_state.handlers, entry) {



        if ((!se->ops || !se->ops->save_state) && !se->vmsd) {

            continue;

        }

        if (se->vmsd && !vmstate_save_needed(se->vmsd, se->opaque)) {

            trace_savevm_section_skip(se->idstr, se->section_id);

            continue;

        }



        trace_savevm_section_start(se->idstr, se->section_id);



        json_start_object(vmdesc, NULL);

        json_prop_str(vmdesc, "name", se->idstr);

        json_prop_int(vmdesc, "instance_id", se->instance_id);



        save_section_header(f, se, QEMU_VM_SECTION_FULL);

        vmstate_save(f, se, vmdesc);

        trace_savevm_section_end(se->idstr, se->section_id, 0);

        save_section_footer(f, se);



        json_end_object(vmdesc);

    }



    if (!in_postcopy) {

        /* Postcopy stream will still be going */

        qemu_put_byte(f, QEMU_VM_EOF);

    }



    json_end_array(vmdesc);

    qjson_finish(vmdesc);

    vmdesc_len = strlen(qjson_get_str(vmdesc));



    if (should_send_vmdesc()) {

        qemu_put_byte(f, QEMU_VM_VMDESCRIPTION);

        qemu_put_be32(f, vmdesc_len);

        qemu_put_buffer(f, (uint8_t *)qjson_get_str(vmdesc), vmdesc_len);

    }

    qjson_destroy(vmdesc);



    qemu_fflush(f);

}
