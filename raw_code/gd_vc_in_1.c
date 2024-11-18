static gboolean gd_vc_in(GIOChannel *chan, GIOCondition cond, void *opaque)

{

    VirtualConsole *vc = opaque;

    uint8_t buffer[1024];

    ssize_t len;



    len = read(vc->fd, buffer, sizeof(buffer));

    if (len <= 0) {

        return FALSE;

    }



    qemu_chr_be_write(vc->chr, buffer, len);



    return TRUE;

}
