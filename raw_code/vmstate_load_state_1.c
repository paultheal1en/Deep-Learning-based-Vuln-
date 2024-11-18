int vmstate_load_state(QEMUFile *f, const VMStateDescription *vmsd,

                       void *opaque, int version_id)

{

    VMStateField *field = vmsd->fields;

    int ret = 0;



    trace_vmstate_load_state(vmsd->name, version_id);

    if (version_id > vmsd->version_id) {

        error_report("%s: incoming version_id %d is too new "

                     "for local version_id %d",

                     vmsd->name, version_id, vmsd->version_id);

        trace_vmstate_load_state_end(vmsd->name, "too new", -EINVAL);

        return -EINVAL;

    }

    if  (version_id < vmsd->minimum_version_id) {

        if (vmsd->load_state_old &&

            version_id >= vmsd->minimum_version_id_old) {

            ret = vmsd->load_state_old(f, opaque, version_id);

            trace_vmstate_load_state_end(vmsd->name, "old path", ret);

            return ret;

        }

        error_report("%s: incoming version_id %d is too old "

                     "for local minimum version_id  %d",

                     vmsd->name, version_id, vmsd->minimum_version_id);

        trace_vmstate_load_state_end(vmsd->name, "too old", -EINVAL);

        return -EINVAL;

    }

    if (vmsd->pre_load) {

        int ret = vmsd->pre_load(opaque);

        if (ret) {

            return ret;

        }

    }

    while (field->name) {

        trace_vmstate_load_state_field(vmsd->name, field->name);

        if ((field->field_exists &&

             field->field_exists(opaque, version_id)) ||

            (!field->field_exists &&

             field->version_id <= version_id)) {

            void *first_elem = opaque + field->offset;

            int i, n_elems = vmstate_n_elems(opaque, field);

            int size = vmstate_size(opaque, field);



            vmstate_handle_alloc(first_elem, field, opaque);

            if (field->flags & VMS_POINTER) {

                first_elem = *(void **)first_elem;

                assert(first_elem  || !n_elems);

            }

            for (i = 0; i < n_elems; i++) {

                void *curr_elem = first_elem + size * i;



                if (field->flags & VMS_ARRAY_OF_POINTER) {

                    curr_elem = *(void **)curr_elem;

                }

                if (field->flags & VMS_STRUCT) {

                    ret = vmstate_load_state(f, field->vmsd, curr_elem,

                                             field->vmsd->version_id);

                } else {

                    ret = field->info->get(f, curr_elem, size, field);

                }

                if (ret >= 0) {

                    ret = qemu_file_get_error(f);

                }

                if (ret < 0) {

                    qemu_file_set_error(f, ret);

                    error_report("Failed to load %s:%s", vmsd->name,

                                 field->name);

                    trace_vmstate_load_field_error(field->name, ret);

                    return ret;

                }

            }

        } else if (field->flags & VMS_MUST_EXIST) {

            error_report("Input validation failed: %s/%s",

                         vmsd->name, field->name);

            return -1;

        }

        field++;

    }

    ret = vmstate_subsection_load(f, vmsd, opaque);

    if (ret != 0) {

        return ret;

    }

    if (vmsd->post_load) {

        ret = vmsd->post_load(opaque, version_id);

    }

    trace_vmstate_load_state_end(vmsd->name, "end", ret);

    return ret;

}
