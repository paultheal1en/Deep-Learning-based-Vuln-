static GSList *gd_vc_vte_init(GtkDisplayState *s, VirtualConsole *vc,

                              CharDriverState *chr, int idx,

                              GSList *group, GtkWidget *view_menu)

{

    char buffer[32];

    GtkWidget *box;

    GtkWidget *scrollbar;

    GtkAdjustment *vadjustment;

    VirtualConsole *tmp_vc = chr->opaque;



    vc->s = s;

    vc->vte.echo = tmp_vc->vte.echo;



    vc->vte.chr = chr;

    chr->opaque = vc;

    g_free(tmp_vc);



    snprintf(buffer, sizeof(buffer), "vc%d", idx);

    vc->label = g_strdup_printf("%s", vc->vte.chr->label

                                ? vc->vte.chr->label : buffer);

    group = gd_vc_menu_init(s, vc, idx, group, view_menu);



    vc->vte.terminal = vte_terminal_new();

    g_signal_connect(vc->vte.terminal, "commit", G_CALLBACK(gd_vc_in), vc);



    /* The documentation says that the default is UTF-8, but actually it is

     * 7-bit ASCII at least in VTE 0.38.

     */

#if VTE_CHECK_VERSION(0, 40, 0)

    vte_terminal_set_encoding(VTE_TERMINAL(vc->vte.terminal), "UTF-8", NULL);

#else

    vte_terminal_set_encoding(VTE_TERMINAL(vc->vte.terminal), "UTF-8");

#endif



    vte_terminal_set_scrollback_lines(VTE_TERMINAL(vc->vte.terminal), -1);

    vte_terminal_set_size(VTE_TERMINAL(vc->vte.terminal),

                          VC_TERM_X_MIN, VC_TERM_Y_MIN);



#if VTE_CHECK_VERSION(0, 28, 0) && GTK_CHECK_VERSION(3, 0, 0)

    vadjustment = gtk_scrollable_get_vadjustment

        (GTK_SCROLLABLE(vc->vte.terminal));

#else

    vadjustment = vte_terminal_get_adjustment(VTE_TERMINAL(vc->vte.terminal));

#endif



#if GTK_CHECK_VERSION(3, 0, 0)

    box = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 2);

    scrollbar = gtk_scrollbar_new(GTK_ORIENTATION_VERTICAL, vadjustment);

#else

    box = gtk_hbox_new(false, 2);

    scrollbar = gtk_vscrollbar_new(vadjustment);

#endif



    gtk_box_pack_start(GTK_BOX(box), vc->vte.terminal, TRUE, TRUE, 0);

    gtk_box_pack_start(GTK_BOX(box), scrollbar, FALSE, FALSE, 0);



    vc->vte.box = box;

    vc->vte.scrollbar = scrollbar;



    g_signal_connect(vadjustment, "changed",

                     G_CALLBACK(gd_vc_adjustment_changed), vc);



    vc->type = GD_VC_VTE;

    vc->tab_item = box;

    vc->focus = vc->vte.terminal;

    gtk_notebook_append_page(GTK_NOTEBOOK(s->notebook), vc->tab_item,

                             gtk_label_new(vc->label));



    qemu_chr_be_generic_open(vc->vte.chr);

    if (vc->vte.chr->init) {

        vc->vte.chr->init(vc->vte.chr);

    }



    return group;

}
