static void vnc_display_close(DisplayState *ds)

{

    VncDisplay *vs = ds ? (VncDisplay *)ds->opaque : vnc_display;



    if (!vs)

        return;

    if (vs->display) {

        g_free(vs->display);

        vs->display = NULL;

    }

    if (vs->lsock != -1) {

        qemu_set_fd_handler2(vs->lsock, NULL, NULL, NULL, NULL);

        close(vs->lsock);

        vs->lsock = -1;

    }

#ifdef CONFIG_VNC_WS

    g_free(vs->ws_display);

    vs->ws_display = NULL;

    if (vs->lwebsock != -1) {

        qemu_set_fd_handler2(vs->lwebsock, NULL, NULL, NULL, NULL);

        close(vs->lwebsock);

        vs->lwebsock = -1;

    }

#endif /* CONFIG_VNC_WS */

    vs->auth = VNC_AUTH_INVALID;

#ifdef CONFIG_VNC_TLS

    vs->subauth = VNC_AUTH_INVALID;

    vs->tls.x509verify = 0;

#endif

}
