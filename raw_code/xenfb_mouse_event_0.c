static void xenfb_mouse_event(void *opaque,

			      int dx, int dy, int dz, int button_state)

{

    struct XenInput *xenfb = opaque;

    DisplaySurface *surface = qemu_console_surface(xenfb->c.con);

    int dw = surface_width(surface);

    int dh = surface_height(surface);

    int i;



    trace_xenfb_mouse_event(opaque, dx, dy, dz, button_state,

                            xenfb->abs_pointer_wanted);

    if (xenfb->abs_pointer_wanted)

	xenfb_send_position(xenfb,

			    dx * (dw - 1) / 0x7fff,

			    dy * (dh - 1) / 0x7fff,

			    dz);

    else

	xenfb_send_motion(xenfb, dx, dy, dz);



    for (i = 0 ; i < 8 ; i++) {

	int lastDown = xenfb->button_state & (1 << i);

	int down = button_state & (1 << i);

	if (down == lastDown)

	    continue;



	if (xenfb_send_key(xenfb, down, BTN_LEFT+i) < 0)

	    return;

    }

    xenfb->button_state = button_state;

}
