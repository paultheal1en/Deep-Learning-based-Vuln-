void vnc_display_close(DisplayState *ds)

{

    VncDisplay *vs = ds ? (VncDisplay *)ds->opaque : vnc_display;



    if (!vs)

        return;

    if (vs->display) {

	qemu_free(vs->display);

	vs->display = NULL;

    }

    if (vs->lsock != -1) {

	qemu_set_fd_handler2(vs->lsock, NULL, NULL, NULL, NULL);

	close(vs->lsock);

	vs->lsock = -1;

    }

    vs->auth = VNC_AUTH_INVALID;

#ifdef CONFIG_VNC_TLS

    vs->subauth = VNC_AUTH_INVALID;

    vs->x509verify = 0;

#endif

}
