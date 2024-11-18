int qemu_get_fd(QEMUFile *f)

{

    if (f->ops->get_fd) {

        return f->ops->get_fd(f->opaque);

    }

    return -1;

}
