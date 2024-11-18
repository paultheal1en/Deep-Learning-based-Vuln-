void gtk_display_init(DisplayState *ds)

{

    GtkDisplayState *s = g_malloc0(sizeof(*s));



    gtk_init(NULL, NULL);



    ds->opaque = s;

    s->ds = ds;

    s->dcl.ops = &dcl_ops;



    s->window = gtk_window_new(GTK_WINDOW_TOPLEVEL);

#if GTK_CHECK_VERSION(3, 2, 0)

    s->vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);

#else

    s->vbox = gtk_vbox_new(FALSE, 0);

#endif

    s->notebook = gtk_notebook_new();

    s->drawing_area = gtk_drawing_area_new();

    s->menu_bar = gtk_menu_bar_new();



    s->scale_x = 1.0;

    s->scale_y = 1.0;

    s->free_scale = FALSE;



    setlocale(LC_ALL, "");

    bindtextdomain("qemu", CONFIG_QEMU_LOCALEDIR);

    textdomain("qemu");



    s->null_cursor = gdk_cursor_new(GDK_BLANK_CURSOR);



    s->mouse_mode_notifier.notify = gd_mouse_mode_change;

    qemu_add_mouse_mode_change_notifier(&s->mouse_mode_notifier);

    qemu_add_vm_change_state_handler(gd_change_runstate, s);



    gtk_notebook_append_page(GTK_NOTEBOOK(s->notebook), s->drawing_area, gtk_label_new("VGA"));



    gd_create_menus(s);



    gd_connect_signals(s);



    gtk_widget_add_events(s->drawing_area,

                          GDK_POINTER_MOTION_MASK |

                          GDK_BUTTON_PRESS_MASK |

                          GDK_BUTTON_RELEASE_MASK |

                          GDK_BUTTON_MOTION_MASK |

                          GDK_ENTER_NOTIFY_MASK |

                          GDK_LEAVE_NOTIFY_MASK |

                          GDK_SCROLL_MASK |

                          GDK_KEY_PRESS_MASK);

    gtk_widget_set_double_buffered(s->drawing_area, FALSE);

    gtk_widget_set_can_focus(s->drawing_area, TRUE);



    gtk_notebook_set_show_tabs(GTK_NOTEBOOK(s->notebook), FALSE);

    gtk_notebook_set_show_border(GTK_NOTEBOOK(s->notebook), FALSE);



    gd_update_caption(s);



    gtk_box_pack_start(GTK_BOX(s->vbox), s->menu_bar, FALSE, TRUE, 0);

    gtk_box_pack_start(GTK_BOX(s->vbox), s->notebook, TRUE, TRUE, 0);



    gtk_container_add(GTK_CONTAINER(s->window), s->vbox);



    gtk_widget_show_all(s->window);



    register_displaychangelistener(ds, &s->dcl);



    global_state = s;

}
