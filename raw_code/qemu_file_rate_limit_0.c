int qemu_file_rate_limit(QEMUFile *f)

{

    if (f->ops->rate_limit)

        return f->ops->rate_limit(f->opaque);



    return 0;

}
