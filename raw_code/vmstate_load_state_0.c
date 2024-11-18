int vmstate_load_state(QEMUFile *f, const VMStateDescription *vmsd,

                       void *opaque, int version_id)

{

    VMStateField *field = vmsd->fields;



    if (version_id > vmsd->version_id) {

        return -EINVAL;

    }

    if (version_id < vmsd->minimum_version_id_old) {

        return -EINVAL;

    }

    if  (version_id < vmsd->minimum_version_id) {

        return vmsd->load_state_old(f, opaque, version_id);

    }

    while(field->name) {

        if (field->version_id <= version_id) {

            void *base_addr = opaque + field->offset;

            int ret, i, n_elems = 1;



            if (field->flags & VMS_ARRAY) {

                n_elems = field->num;

            } else if (field->flags & VMS_VARRAY) {

                n_elems = *(size_t *)(opaque+field->num_offset);

            }

            if (field->flags & VMS_POINTER) {

                base_addr = *(void **)base_addr;

            }

            for (i = 0; i < n_elems; i++) {

                void *addr = base_addr + field->size * i;



                if (field->flags & VMS_STRUCT) {

                    ret = vmstate_load_state(f, field->vmsd, addr, version_id);

                } else {

                    ret = field->info->get(f, addr, field->size);



                }

                if (ret < 0) {

                    return ret;

                }

            }

        }

        field++;

    }

    if (vmsd->run_after_load)

        return vmsd->run_after_load(opaque);

    return 0;

}
