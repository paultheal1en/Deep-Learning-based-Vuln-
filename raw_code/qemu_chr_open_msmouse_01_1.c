static CharDriverState *qemu_chr_open_msmouse(const char *id,

                                              ChardevBackend *backend,

                                              ChardevReturn *ret,

                                              Error **errp)

{

    ChardevCommon *common = backend->u.msmouse.data;

    MouseState *mouse;

    CharDriverState *chr;



    chr = qemu_chr_alloc(common, errp);




    chr->chr_write = msmouse_chr_write;

    chr->chr_close = msmouse_chr_close;

    chr->chr_accept_input = msmouse_chr_accept_input;

    chr->explicit_be_open = true;



    mouse = g_new0(MouseState, 1);

    mouse->hs = qemu_input_handler_register((DeviceState *)mouse,

                                            &msmouse_handler);



    mouse->chr = chr;

    chr->opaque = mouse;



    return chr;
