int64_t qemu_ftell(QEMUFile *f)

{

    qemu_fflush(f);

    return f->pos;

}
