dispatcher_wait(Dispatcher *dispr, uint32_t timeout)

{

    struct timeval tv;

    tv.tv_sec = timeout / 1000000;

    tv.tv_usec = timeout % 1000000;



    fd_set fdset = dispr->fdset;



    /* wait until some of sockets become readable. */

    int rc = select(dispr->max_sock + 1, &fdset, 0, 0, &tv);



    if (rc == -1) {

        vubr_die("select");

    }



    /* Timeout */

    if (rc == 0) {

        return 0;

    }



    /* Now call callback for every ready socket. */



    int sock;

    for (sock = 0; sock < dispr->max_sock + 1; sock++)

        if (FD_ISSET(sock, &fdset)) {

            Event *e = &dispr->events[sock];

            e->callback(sock, e->ctx);

        }



    return 0;

}
