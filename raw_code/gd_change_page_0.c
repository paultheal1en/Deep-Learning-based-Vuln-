static void gd_change_page(GtkNotebook *nb, gpointer arg1, guint arg2,

                           gpointer data)

{

    GtkDisplayState *s = data;

    VirtualConsole *vc;

    gboolean on_vga;



    if (!gtk_widget_get_realized(s->notebook)) {

        return;

    }



    vc = gd_vc_find_by_page(s, arg2);

    if (!vc) {

        return;

    }

    gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(vc->menu_item),

                                   TRUE);



    on_vga = (vc->type == GD_VC_GFX);

    if (!on_vga) {

        gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(s->grab_item),

                                       FALSE);

    } else if (s->full_screen) {

        gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(s->grab_item),

                                       TRUE);

    }

    gtk_widget_set_sensitive(s->grab_item, on_vga);



    gd_update_cursor(vc);

}
