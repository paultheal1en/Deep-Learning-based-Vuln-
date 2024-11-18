int64_t qemu_file_get_rate_limit(QEMUFile *f)

{

    if (f->ops->get_rate_limit)

        return f->ops->get_rate_limit(f->opaque);



    return 0;

}
