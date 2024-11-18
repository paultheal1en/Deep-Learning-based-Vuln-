void ram_control_after_iterate(QEMUFile *f, uint64_t flags)

{

    int ret = 0;



    if (f->ops->after_ram_iterate) {

        ret = f->ops->after_ram_iterate(f, f->opaque, flags);

        if (ret < 0) {

            qemu_file_set_error(f, ret);

        }

    }

}
