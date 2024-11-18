void qemu_savevm_state_complete(QEMUFile *f)

{

    QJSON *vmdesc;

    int vmdesc_len;

    SaveStateEntry *se;

    int ret;



    trace_savevm_state_complete();



    cpu_synchronize_all_states();



    QTAILQ_FOREACH(se, &savevm_state.handlers, entry) {

        if (!se->ops || !se->ops->save_live_complete) {

            continue;

        }

        if (se->ops && se->ops->is_active) {

            if (!se->ops->is_active(se->opaque)) {

                continue;

            }

        }

        trace_savevm_section_start(se->idstr, se->section_id);



        save_section_header(f, se, QEMU_VM_SECTION_END);



        ret = se->ops->save_live_complete(f, se->opaque);

        trace_savevm_section_end(se->idstr, se->section_id, ret);


        if (ret < 0) {

            qemu_file_set_error(f, ret);

            return;

        }

    }



    vmdesc = qjson_new();

    json_prop_int(vmdesc, "page_size", TARGET_PAGE_SIZE);

    json_start_array(vmdesc, "devices");

    QTAILQ_FOREACH(se, &savevm_state.handlers, entry) {



        if ((!se->ops || !se->ops->save_state) && !se->vmsd) {

            continue;

        }

        trace_savevm_section_start(se->idstr, se->section_id);



        json_start_object(vmdesc, NULL);

        json_prop_str(vmdesc, "name", se->idstr);

        json_prop_int(vmdesc, "instance_id", se->instance_id);



        save_section_header(f, se, QEMU_VM_SECTION_FULL);



        vmstate_save(f, se, vmdesc);



        json_end_object(vmdesc);

        trace_savevm_section_end(se->idstr, se->section_id, 0);


    }



    qemu_put_byte(f, QEMU_VM_EOF);



    json_end_array(vmdesc);

    qjson_finish(vmdesc);

    vmdesc_len = strlen(qjson_get_str(vmdesc));



    if (should_send_vmdesc()) {

        qemu_put_byte(f, QEMU_VM_VMDESCRIPTION);

        qemu_put_be32(f, vmdesc_len);

        qemu_put_buffer(f, (uint8_t *)qjson_get_str(vmdesc), vmdesc_len);

    }

    object_unref(OBJECT(vmdesc));



    qemu_fflush(f);

}