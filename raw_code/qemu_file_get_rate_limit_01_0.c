size_t qemu_file_get_rate_limit(QEMUFile *f)

{

    if (f->get_rate_limit)

        return f->get_rate_limit(f->opaque);



    return 0;

}
