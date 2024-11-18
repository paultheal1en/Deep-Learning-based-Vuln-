void vnc_display_init(const char *id)

{

    VncDisplay *vs;



    if (vnc_display_find(id) != NULL) {

        return;

    }

    vs = g_malloc0(sizeof(*vs));



    vs->id = strdup(id);

    QTAILQ_INSERT_TAIL(&vnc_displays, vs, next);



    vs->lsock = -1;

#ifdef CONFIG_VNC_WS

    vs->lwebsock = -1;

#endif



    QTAILQ_INIT(&vs->clients);

    vs->expires = TIME_MAX;



    if (keyboard_layout) {

        trace_vnc_key_map_init(keyboard_layout);

        vs->kbd_layout = init_keyboard_layout(name2keysym, keyboard_layout);

    } else {

        vs->kbd_layout = init_keyboard_layout(name2keysym, "en-us");

    }



    if (!vs->kbd_layout)

        exit(1);



    qemu_mutex_init(&vs->mutex);

    vnc_start_worker_thread();



    vs->dcl.ops = &dcl_ops;

    register_displaychangelistener(&vs->dcl);

}
