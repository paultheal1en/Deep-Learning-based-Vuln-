bool qemu_file_is_writable(QEMUFile *f)

{

    return f->ops->writev_buffer || f->ops->put_buffer;

}
