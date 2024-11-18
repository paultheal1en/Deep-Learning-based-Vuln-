int qemu_chr_open_msmouse(QemuOpts *opts, CharDriverState **_chr)

{

    CharDriverState *chr;



    chr = g_malloc0(sizeof(CharDriverState));

    chr->chr_write = msmouse_chr_write;

    chr->chr_close = msmouse_chr_close;



    qemu_add_mouse_event_handler(msmouse_event, chr, 0, "QEMU Microsoft Mouse");



    *_chr = chr;

    return 0;

}
