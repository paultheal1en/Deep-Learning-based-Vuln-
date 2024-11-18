void vnc_display_init(DisplayState *ds)

{

    VncState *vs;



    vs = qemu_mallocz(sizeof(VncState));

    if (!vs)

	exit(1);



    ds->opaque = vs;

    vnc_state = vs;

    vs->display = NULL;

    vs->password = NULL;



    vs->lsock = -1;

    vs->csock = -1;

    vs->depth = 4;

    vs->last_x = -1;

    vs->last_y = -1;



    vs->ds = ds;



    if (!keyboard_layout)

	keyboard_layout = "en-us";



    vs->kbd_layout = init_keyboard_layout(keyboard_layout);

    if (!vs->kbd_layout)

	exit(1);



    vs->timer = qemu_new_timer(rt_clock, vnc_update_client, vs);



    vs->ds->data = NULL;

    vs->ds->dpy_update = vnc_dpy_update;

    vs->ds->dpy_resize = vnc_dpy_resize;

    vs->ds->dpy_refresh = NULL;



    memset(vs->dirty_row, 0xFF, sizeof(vs->dirty_row));



    vnc_dpy_resize(vs->ds, 640, 400);

}
