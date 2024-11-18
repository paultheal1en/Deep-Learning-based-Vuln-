void qemu_file_set_error(QEMUFile *f, int ret)

{

    if (f->last_error == 0) {

        f->last_error = ret;

    }

}
