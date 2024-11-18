static GSList *gd_vc_gfx_init(GtkDisplayState *s, VirtualConsole *vc,

                              QemuConsole *con, int idx,

                              GSList *group, GtkWidget *view_menu)

{

    Error *local_err = NULL;

    Object *obj;



    obj = object_property_get_link(OBJECT(con), "device", &local_err);

    if (obj) {

        vc->label = g_strdup_printf("%s", object_get_typename(obj));

    } else {

        vc->label = g_strdup_printf("VGA");

    }



    vc->s = s;

    vc->gfx.scale_x = 1.0;

    vc->gfx.scale_y = 1.0;



    vc->gfx.drawing_area = gtk_drawing_area_new();

    gtk_widget_add_events(vc->gfx.drawing_area,

                          GDK_POINTER_MOTION_MASK |

                          GDK_BUTTON_PRESS_MASK |

                          GDK_BUTTON_RELEASE_MASK |

                          GDK_BUTTON_MOTION_MASK |

                          GDK_ENTER_NOTIFY_MASK |

                          GDK_LEAVE_NOTIFY_MASK |

                          GDK_SCROLL_MASK |

                          GDK_KEY_PRESS_MASK);

    gtk_widget_set_can_focus(vc->gfx.drawing_area, TRUE);



    vc->type = GD_VC_GFX;

    vc->tab_item = vc->gfx.drawing_area;

    gtk_notebook_append_page(GTK_NOTEBOOK(s->notebook),

                             vc->tab_item, gtk_label_new(vc->label));

    gd_connect_vc_gfx_signals(vc);



    group = gd_vc_menu_init(s, vc, idx, group, view_menu);



    vc->gfx.dcl.ops = &dcl_ops;

    vc->gfx.dcl.con = con;

    register_displaychangelistener(&vc->gfx.dcl);



    return group;

}
