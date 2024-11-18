static gboolean gd_focus_out_event(GtkWidget *widget,

                                   GdkEventCrossing *crossing, gpointer opaque)

{

    VirtualConsole *vc = opaque;

    GtkDisplayState *s = vc->s;



    gtk_release_modifiers(s);



    return TRUE;

}
