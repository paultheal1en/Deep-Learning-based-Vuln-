static gboolean gd_scroll_event(GtkWidget *widget, GdkEventScroll *scroll,

                                void *opaque)

{

    VirtualConsole *vc = opaque;

    InputButton btn;



    if (scroll->direction == GDK_SCROLL_UP) {

        btn = INPUT_BUTTON_WHEEL_UP;

    } else if (scroll->direction == GDK_SCROLL_DOWN) {

        btn = INPUT_BUTTON_WHEEL_DOWN;

    } else {

        return TRUE;

    }



    qemu_input_queue_btn(vc->gfx.dcl.con, btn, true);

    qemu_input_event_sync();

    qemu_input_queue_btn(vc->gfx.dcl.con, btn, false);

    qemu_input_event_sync();

    return TRUE;

}
