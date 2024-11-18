static int buffered_get_fd(void *opaque)

{

    QEMUFileBuffered *s = opaque;



    return qemu_get_fd(s->file);

}
