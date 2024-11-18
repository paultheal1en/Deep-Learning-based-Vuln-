void vnc_display_init(DisplayState *ds)

{

    VncDisplay *vs = g_malloc0(sizeof(*vs));



    dcl = g_malloc0(sizeof(DisplayChangeListener));



    ds->opaque = vs;

    dcl->idle = 1;

    vnc_display = vs;



    vs->lsock = -1;

#ifdef CONFIG_VNC_WS

    vs->lwebsock = -1;

#endif



    vs->ds = ds;

    QTAILQ_INIT(&vs->clients);

    vs->expires = TIME_MAX;



    if (keyboard_layout)

        vs->kbd_layout = init_keyboard_layout(name2keysym, keyboard_layout);

    else

        vs->kbd_layout = init_keyboard_layout(name2keysym, "en-us");



    if (!vs->kbd_layout)

        exit(1);



    qemu_mutex_init(&vs->mutex);

    vnc_start_worker_thread();



    dcl->ops = &dcl_ops;

    register_displaychangelistener(ds, dcl);

}
