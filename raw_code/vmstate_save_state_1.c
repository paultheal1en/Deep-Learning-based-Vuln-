void vmstate_save_state(QEMUFile *f, const VMStateDescription *vmsd,

                        void *opaque)

{

    VMStateField *field = vmsd->fields;



    if (vmsd->pre_save) {

        vmsd->pre_save(opaque);


    while (field->name) {

        if (!field->field_exists ||

            field->field_exists(opaque, vmsd->version_id)) {

            void *base_addr = vmstate_base_addr(opaque, field);

            int i, n_elems = vmstate_n_elems(opaque, field);

            int size = vmstate_size(opaque, field);



            for (i = 0; i < n_elems; i++) {

                void *addr = base_addr + size * i;



                if (field->flags & VMS_ARRAY_OF_POINTER) {

                    addr = *(void **)addr;


                if (field->flags & VMS_STRUCT) {

                    vmstate_save_state(f, field->vmsd, addr);


                    field->info->put(f, addr, size);










        field++;


    vmstate_subsection_save(f, vmsd, opaque);
