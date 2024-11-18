static void gd_update_cursor(GtkDisplayState *s, gboolean override)

{

    GdkWindow *window;

    bool on_vga;



    window = gtk_widget_get_window(GTK_WIDGET(s->drawing_area));



    on_vga = (gtk_notebook_get_current_page(GTK_NOTEBOOK(s->notebook)) == 0);



    if ((override || on_vga) && kbd_mouse_is_absolute()) {

        gdk_window_set_cursor(window, s->null_cursor);

    } else {

        gdk_window_set_cursor(window, NULL);

    }

}
