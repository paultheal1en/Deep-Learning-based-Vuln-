void vmstate_save_state(QEMUFile *f, const VMStateDescription *vmsd,

                        void *opaque, QJSON *vmdesc)

{

    VMStateField *field = vmsd->fields;



    trace_vmstate_save_state_top(vmsd->name);



    if (vmsd->pre_save) {

        vmsd->pre_save(opaque);

    }



    if (vmdesc) {

        json_prop_str(vmdesc, "vmsd_name", vmsd->name);

        json_prop_int(vmdesc, "version", vmsd->version_id);

        json_start_array(vmdesc, "fields");

    }



    while (field->name) {

        if (!field->field_exists ||

            field->field_exists(opaque, vmsd->version_id)) {

            void *first_elem = opaque + field->offset;

            int i, n_elems = vmstate_n_elems(opaque, field);

            int size = vmstate_size(opaque, field);

            int64_t old_offset, written_bytes;

            QJSON *vmdesc_loop = vmdesc;



            trace_vmstate_save_state_loop(vmsd->name, field->name, n_elems);

            if (field->flags & VMS_POINTER) {

                first_elem = *(void **)first_elem;

                assert(first_elem  || !n_elems);

            }

            for (i = 0; i < n_elems; i++) {

                void *curr_elem = first_elem + size * i;



                vmsd_desc_field_start(vmsd, vmdesc_loop, field, i, n_elems);

                old_offset = qemu_ftell_fast(f);

                if (field->flags & VMS_ARRAY_OF_POINTER) {

                    assert(curr_elem);

                    curr_elem = *(void **)curr_elem;

                }

                if (field->flags & VMS_STRUCT) {

                    vmstate_save_state(f, field->vmsd, curr_elem, vmdesc_loop);

                } else {

                    field->info->put(f, curr_elem, size, field, vmdesc_loop);

                }



                written_bytes = qemu_ftell_fast(f) - old_offset;

                vmsd_desc_field_end(vmsd, vmdesc_loop, field, written_bytes, i);



                /* Compressed arrays only care about the first element */

                if (vmdesc_loop && vmsd_can_compress(field)) {

                    vmdesc_loop = NULL;

                }

            }

        } else {

            if (field->flags & VMS_MUST_EXIST) {

                error_report("Output state validation failed: %s/%s",

                        vmsd->name, field->name);

                assert(!(field->flags & VMS_MUST_EXIST));

            }

        }

        field++;

    }



    if (vmdesc) {

        json_end_array(vmdesc);

    }



    vmstate_subsection_save(f, vmsd, opaque, vmdesc);

}
