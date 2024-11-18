void ram_control_load_hook(QEMUFile *f, uint64_t flags)

{

    int ret = -EINVAL;



    if (f->ops->hook_ram_load) {

        ret = f->ops->hook_ram_load(f, f->opaque, flags);

        if (ret < 0) {

            qemu_file_set_error(f, ret);

        }

    } else {

        qemu_file_set_error(f, ret);

    }

}
