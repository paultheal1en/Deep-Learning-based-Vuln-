static void gd_grab_pointer(GtkDisplayState *s)

{

#if GTK_CHECK_VERSION(3, 0, 0)

    GdkDisplay *display = gtk_widget_get_display(s->drawing_area);

    GdkDeviceManager *mgr = gdk_display_get_device_manager(display);

    GList *devices = gdk_device_manager_list_devices(mgr,

                                                     GDK_DEVICE_TYPE_MASTER);

    GList *tmp = devices;

    while (tmp) {

        GdkDevice *dev = tmp->data;

        if (gdk_device_get_source(dev) == GDK_SOURCE_MOUSE) {

            gdk_device_grab(dev,

                            gtk_widget_get_window(s->drawing_area),

                            GDK_OWNERSHIP_NONE,

                            FALSE, /* All events to come to our

                                      window directly */

                            GDK_POINTER_MOTION_MASK |

                            GDK_BUTTON_PRESS_MASK |

                            GDK_BUTTON_RELEASE_MASK |

                            GDK_BUTTON_MOTION_MASK |

                            GDK_SCROLL_MASK,

                            s->null_cursor,

                            GDK_CURRENT_TIME);

        }

        tmp = tmp->next;

    }

    g_list_free(devices);

#else

    gdk_pointer_grab(gtk_widget_get_window(s->drawing_area),

                     FALSE, /* All events to come to our window directly */

                     GDK_POINTER_MOTION_MASK |

                     GDK_BUTTON_PRESS_MASK |

                     GDK_BUTTON_RELEASE_MASK |

                     GDK_BUTTON_MOTION_MASK |

                     GDK_SCROLL_MASK,

                     NULL, /* Allow cursor to move over entire desktop */

                     s->null_cursor,

                     GDK_CURRENT_TIME);

#endif

}
