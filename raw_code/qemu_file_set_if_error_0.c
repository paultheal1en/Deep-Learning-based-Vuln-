static void qemu_file_set_if_error(QEMUFile *f, int ret)

{

    if (ret < 0 && !f->last_error) {

        qemu_file_set_error(f, ret);

    }

}
