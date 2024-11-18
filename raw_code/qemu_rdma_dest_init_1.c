static int qemu_rdma_dest_init(RDMAContext *rdma, Error **errp)

{

    int ret = -EINVAL, idx;

    struct rdma_cm_id *listen_id;

    char ip[40] = "unknown";

    struct addrinfo *res;

    char port_str[16];



    for (idx = 0; idx < RDMA_WRID_MAX; idx++) {

        rdma->wr_data[idx].control_len = 0;

        rdma->wr_data[idx].control_curr = NULL;

    }



    if (rdma->host == NULL) {

        ERROR(errp, "RDMA host is not set!");

        rdma->error_state = -EINVAL;

        return -1;

    }

    /* create CM channel */

    rdma->channel = rdma_create_event_channel();

    if (!rdma->channel) {

        ERROR(errp, "could not create rdma event channel");

        rdma->error_state = -EINVAL;

        return -1;

    }



    /* create CM id */

    ret = rdma_create_id(rdma->channel, &listen_id, NULL, RDMA_PS_TCP);

    if (ret) {

        ERROR(errp, "could not create cm_id!");

        goto err_dest_init_create_listen_id;

    }



    snprintf(port_str, 16, "%d", rdma->port);

    port_str[15] = '\0';



    if (rdma->host && strcmp("", rdma->host)) {

        struct addrinfo *e;



        ret = getaddrinfo(rdma->host, port_str, NULL, &res);

        if (ret < 0) {

            ERROR(errp, "could not getaddrinfo address %s", rdma->host);

            goto err_dest_init_bind_addr;

        }



        for (e = res; e != NULL; e = e->ai_next) {

            inet_ntop(e->ai_family,

                &((struct sockaddr_in *) e->ai_addr)->sin_addr, ip, sizeof ip);

            DPRINTF("Trying %s => %s\n", rdma->host, ip);

            ret = rdma_bind_addr(listen_id, e->ai_addr);

            if (!ret) {

                goto listen;

            }

        }



        ERROR(errp, "Error: could not rdma_bind_addr!");

        goto err_dest_init_bind_addr;

    } else {

        ERROR(errp, "migration host and port not specified!");

        ret = -EINVAL;

        goto err_dest_init_bind_addr;

    }

listen:



    rdma->listen_id = listen_id;

    qemu_rdma_dump_gid("dest_init", listen_id);

    return 0;



err_dest_init_bind_addr:

    rdma_destroy_id(listen_id);

err_dest_init_create_listen_id:

    rdma_destroy_event_channel(rdma->channel);

    rdma->channel = NULL;

    rdma->error_state = ret;

    return ret;



}
