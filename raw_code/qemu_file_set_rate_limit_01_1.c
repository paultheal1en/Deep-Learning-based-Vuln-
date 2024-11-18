void qemu_file_set_rate_limit(QEMUFile *f, int64_t limit)

{

    f->xfer_limit = limit;

}
