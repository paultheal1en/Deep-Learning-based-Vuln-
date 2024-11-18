static CharDriverState *qemu_chr_open_msmouse(const char *id,

                                              ChardevBackend *backend,

                                              ChardevReturn *ret,

                                              Error **errp)

{

    CharDriverState *chr;



    chr = qemu_chr_alloc();

    chr->chr_write = msmouse_chr_write;

    chr->chr_close = msmouse_chr_close;

    chr->explicit_be_open = true;



    qemu_add_mouse_event_handler(msmouse_event, chr, 0, "QEMU Microsoft Mouse");



    return chr;

}
