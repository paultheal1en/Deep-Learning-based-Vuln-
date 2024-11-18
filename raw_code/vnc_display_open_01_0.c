int vnc_display_open(DisplayState *ds, const char *arg)

{

    struct sockaddr *addr;

    struct sockaddr_in iaddr;

#ifndef _WIN32

    struct sockaddr_un uaddr;

#endif

    int reuse_addr, ret;

    socklen_t addrlen;

    const char *p;

    VncState *vs = ds ? (VncState *)ds->opaque : vnc_state;



    vnc_display_close(ds);

    if (strcmp(arg, "none") == 0)

	return 0;



    if (!(vs->display = strdup(arg)))

	return -1;

#ifndef _WIN32

    if (strstart(arg, "unix:", &p)) {

	addr = (struct sockaddr *)&uaddr;

	addrlen = sizeof(uaddr);



	vs->lsock = socket(PF_UNIX, SOCK_STREAM, 0);

	if (vs->lsock == -1) {

	    fprintf(stderr, "Could not create socket\n");

	    free(vs->display);

	    vs->display = NULL;

	    return -1;

	}



	uaddr.sun_family = AF_UNIX;

	memset(uaddr.sun_path, 0, 108);

	snprintf(uaddr.sun_path, 108, "%s", p);



	unlink(uaddr.sun_path);

    } else

#endif

    {

	addr = (struct sockaddr *)&iaddr;

	addrlen = sizeof(iaddr);



	if (parse_host_port(&iaddr, arg) < 0) {

	    fprintf(stderr, "Could not parse VNC address\n");

	    free(vs->display);

	    vs->display = NULL;

	    return -1;

	}



	iaddr.sin_port = htons(ntohs(iaddr.sin_port) + 5900);



	vs->lsock = socket(PF_INET, SOCK_STREAM, 0);

	if (vs->lsock == -1) {

	    fprintf(stderr, "Could not create socket\n");

	    free(vs->display);

	    vs->display = NULL;

	    return -1;

	}



	reuse_addr = 1;

	ret = setsockopt(vs->lsock, SOL_SOCKET, SO_REUSEADDR,

			 (const char *)&reuse_addr, sizeof(reuse_addr));

	if (ret == -1) {

	    fprintf(stderr, "setsockopt() failed\n");

	    close(vs->lsock);

	    vs->lsock = -1;

	    free(vs->display);

	    vs->display = NULL;

	    return -1;

	}

    }



    if (bind(vs->lsock, addr, addrlen) == -1) {

	fprintf(stderr, "bind() failed\n");

	close(vs->lsock);

	vs->lsock = -1;

	free(vs->display);

	vs->display = NULL;

	return -1;

    }



    if (listen(vs->lsock, 1) == -1) {

	fprintf(stderr, "listen() failed\n");

	close(vs->lsock);

	vs->lsock = -1;

	free(vs->display);

	vs->display = NULL;

	return -1;

    }



    return qemu_set_fd_handler2(vs->lsock, vnc_listen_poll, vnc_listen_read, NULL, vs);

}
