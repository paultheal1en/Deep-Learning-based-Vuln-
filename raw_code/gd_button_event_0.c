static gboolean gd_button_event(GtkWidget *widget, GdkEventButton *button,

                                void *opaque)

{

    VirtualConsole *vc = opaque;

    GtkDisplayState *s = vc->s;

    InputButton btn;



    /* implicitly grab the input at the first click in the relative mode */

    if (button->button == 1 && button->type == GDK_BUTTON_PRESS &&

        !qemu_input_is_absolute() && !gd_is_grab_active(s)) {

        gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(s->grab_item),

                                       TRUE);

        return TRUE;

    }



    if (button->button == 1) {

        btn = INPUT_BUTTON_LEFT;

    } else if (button->button == 2) {

        btn = INPUT_BUTTON_MIDDLE;

    } else if (button->button == 3) {

        btn = INPUT_BUTTON_RIGHT;

    } else {

        return TRUE;

    }



    qemu_input_queue_btn(vc->gfx.dcl.con, btn,

                         button->type == GDK_BUTTON_PRESS);

    qemu_input_event_sync();

    return TRUE;

}
