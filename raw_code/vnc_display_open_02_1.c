int vnc_display_open(DisplayState *ds, const char *display)

{

    VncDisplay *vs = ds ? (VncDisplay *)ds->opaque : vnc_display;

    const char *options;

    int password = 0;

    int reverse = 0;

    int to_port = 0;

#ifdef CONFIG_VNC_TLS

    int tls = 0, x509 = 0;

#endif



    if (!vnc_display)

        return -1;

    vnc_display_close(ds);

    if (strcmp(display, "none") == 0)

	return 0;



    if (!(vs->display = strdup(display)))

	return -1;



    options = display;

    while ((options = strchr(options, ','))) {

	options++;

	if (strncmp(options, "password", 8) == 0) {

	    password = 1; /* Require password auth */

	} else if (strncmp(options, "reverse", 7) == 0) {

	    reverse = 1;

	} else if (strncmp(options, "to=", 3) == 0) {

            to_port = atoi(options+3) + 5900;

#ifdef CONFIG_VNC_TLS

	} else if (strncmp(options, "tls", 3) == 0) {

	    tls = 1; /* Require TLS */

	} else if (strncmp(options, "x509", 4) == 0) {

	    char *start, *end;

	    x509 = 1; /* Require x509 certificates */

	    if (strncmp(options, "x509verify", 10) == 0)

	        vs->tls.x509verify = 1; /* ...and verify client certs */



	    /* Now check for 'x509=/some/path' postfix

	     * and use that to setup x509 certificate/key paths */

	    start = strchr(options, '=');

	    end = strchr(options, ',');

	    if (start && (!end || (start < end))) {

		int len = end ? end-(start+1) : strlen(start+1);

		char *path = qemu_strndup(start + 1, len);



		VNC_DEBUG("Trying certificate path '%s'\n", path);

		if (vnc_tls_set_x509_creds_dir(vs, path) < 0) {

		    fprintf(stderr, "Failed to find x509 certificates/keys in %s\n", path);

		    qemu_free(path);

		    qemu_free(vs->display);

		    vs->display = NULL;

		    return -1;

		}

		qemu_free(path);

	    } else {

		fprintf(stderr, "No certificate path provided\n");

		qemu_free(vs->display);

		vs->display = NULL;

		return -1;

	    }

#endif

	}

    }



    if (password) {

#ifdef CONFIG_VNC_TLS

	if (tls) {

	    vs->auth = VNC_AUTH_VENCRYPT;

	    if (x509) {

		VNC_DEBUG("Initializing VNC server with x509 password auth\n");

		vs->subauth = VNC_AUTH_VENCRYPT_X509VNC;

	    } else {

		VNC_DEBUG("Initializing VNC server with TLS password auth\n");

		vs->subauth = VNC_AUTH_VENCRYPT_TLSVNC;

	    }

	} else {

#endif

	    VNC_DEBUG("Initializing VNC server with password auth\n");

	    vs->auth = VNC_AUTH_VNC;

#ifdef CONFIG_VNC_TLS

	    vs->subauth = VNC_AUTH_INVALID;

	}

#endif

    } else {

#ifdef CONFIG_VNC_TLS

	if (tls) {

	    vs->auth = VNC_AUTH_VENCRYPT;

	    if (x509) {

		VNC_DEBUG("Initializing VNC server with x509 no auth\n");

		vs->subauth = VNC_AUTH_VENCRYPT_X509NONE;

	    } else {

		VNC_DEBUG("Initializing VNC server with TLS no auth\n");

		vs->subauth = VNC_AUTH_VENCRYPT_TLSNONE;

	    }

	} else {

#endif

	    VNC_DEBUG("Initializing VNC server with no auth\n");

	    vs->auth = VNC_AUTH_NONE;

#ifdef CONFIG_VNC_TLS

	    vs->subauth = VNC_AUTH_INVALID;

	}

#endif

    }



    if (reverse) {

        /* connect to viewer */

        if (strncmp(display, "unix:", 5) == 0)

            vs->lsock = unix_connect(display+5);

        else

            vs->lsock = inet_connect(display, SOCK_STREAM);

        if (-1 == vs->lsock) {

            free(vs->display);

            vs->display = NULL;

            return -1;

        } else {

            int csock = vs->lsock;

            vs->lsock = -1;

            vnc_connect(vs, csock);

        }

        return 0;



    } else {

        /* listen for connects */

        char *dpy;

        dpy = qemu_malloc(256);

        if (strncmp(display, "unix:", 5) == 0) {

            pstrcpy(dpy, 256, "unix:");

            vs->lsock = unix_listen(display+5, dpy+5, 256-5);

        } else {

            vs->lsock = inet_listen(display, dpy, 256, SOCK_STREAM, 5900);

        }

        if (-1 == vs->lsock) {

            free(dpy);

            return -1;

        } else {

            free(vs->display);

            vs->display = dpy;

        }

    }

    return qemu_set_fd_handler2(vs->lsock, NULL, vnc_listen_read, NULL, vs);

}
