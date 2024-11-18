static void gd_update_cursor(VirtualConsole *vc)

{

    GtkDisplayState *s = vc->s;

    GdkWindow *window;



    if (vc->type != GD_VC_GFX) {

        return;

    }



    window = gtk_widget_get_window(GTK_WIDGET(vc->gfx.drawing_area));

    if (s->full_screen || qemu_input_is_absolute() || gd_is_grab_active(s)) {

        gdk_window_set_cursor(window, s->null_cursor);

    } else {

        gdk_window_set_cursor(window, NULL);

    }

}
