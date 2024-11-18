static av_cold int xcbgrab_read_header(AVFormatContext *s)

{

    XCBGrabContext *c = s->priv_data;

    int screen_num, ret;

    const xcb_setup_t *setup;

    char *host        = s->filename[0] ? s->filename : NULL;

    const char *opts  = strchr(s->filename, '+');



    if (opts) {

        sscanf(opts, "%d,%d", &c->x, &c->y);

        host = av_strdup(s->filename);

        host[opts - s->filename] = '\0';

    }



    c->conn = xcb_connect(host, &screen_num);



    if (opts)

        av_free(host);



    if ((ret = xcb_connection_has_error(c->conn))) {

        av_log(s, AV_LOG_ERROR, "Cannot open display %s, error %d.\n",

               s->filename[0] ? host : "default", ret);

        return AVERROR(EIO);

    }

    setup = xcb_get_setup(c->conn);



    c->screen = get_screen(setup, screen_num);

    if (!c->screen) {

        av_log(s, AV_LOG_ERROR, "The screen %d does not exist.\n",

               screen_num);

        xcbgrab_read_close(s);

        return AVERROR(EIO);

    }



    ret = create_stream(s);



    if (ret < 0) {

        xcbgrab_read_close(s);

        return ret;

    }



#if CONFIG_LIBXCB_SHM

    if ((c->has_shm = check_shm(c->conn)))

        c->segment = xcb_generate_id(c->conn);

#endif



#if CONFIG_LIBXCB_XFIXES

    if (c->draw_mouse) {

        if (!(c->draw_mouse = check_xfixes(c->conn))) {

            av_log(s, AV_LOG_WARNING,

                   "XFixes not available, cannot draw the mouse.\n");

        }

        if (c->bpp < 24) {

            avpriv_report_missing_feature(s, "%d bits per pixel screen",

                                          c->bpp);

            c->draw_mouse = 0;

        }

    }

#endif



    if (c->show_region)

        setup_window(s);



    return 0;

}
