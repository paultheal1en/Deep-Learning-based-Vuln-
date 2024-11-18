static int check_bind(const char *hostname, bool *has_proto)

{

    int fd = -1;

    struct addrinfo ai, *res = NULL;

    int rc;

    int ret = -1;



    memset(&ai, 0, sizeof(ai));

    ai.ai_flags = AI_CANONNAME | AI_V4MAPPED | AI_ADDRCONFIG;

    ai.ai_family = AF_UNSPEC;

    ai.ai_socktype = SOCK_STREAM;



    /* lookup */

    rc = getaddrinfo(hostname, NULL, &ai, &res);

    if (rc != 0) {

        if (rc == EAI_ADDRFAMILY ||

            rc == EAI_FAMILY) {

            *has_proto = false;

            goto done;

        }

        goto cleanup;

    }



    fd = qemu_socket(res->ai_family, res->ai_socktype, res->ai_protocol);

    if (fd < 0) {

        goto cleanup;

    }



    if (bind(fd, res->ai_addr, res->ai_addrlen) < 0) {

        if (errno == EADDRNOTAVAIL) {

            *has_proto = false;

            goto done;

        }

        goto cleanup;

    }



    *has_proto = true;

 done:

    ret = 0;



 cleanup:

    if (fd != -1) {

        close(fd);

    }

    if (res) {

        freeaddrinfo(res);

    }

    return ret;

}
