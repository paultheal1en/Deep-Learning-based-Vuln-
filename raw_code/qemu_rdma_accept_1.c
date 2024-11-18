static int qemu_rdma_accept(RDMAContext *rdma)

{

    RDMACapabilities cap;

    struct rdma_conn_param conn_param = {

                                            .responder_resources = 2,

                                            .private_data = &cap,

                                            .private_data_len = sizeof(cap),

                                         };

    struct rdma_cm_event *cm_event;

    struct ibv_context *verbs;

    int ret = -EINVAL;

    int idx;



    ret = rdma_get_cm_event(rdma->channel, &cm_event);

    if (ret) {

        goto err_rdma_dest_wait;

    }



    if (cm_event->event != RDMA_CM_EVENT_CONNECT_REQUEST) {

        rdma_ack_cm_event(cm_event);

        goto err_rdma_dest_wait;

    }



    memcpy(&cap, cm_event->param.conn.private_data, sizeof(cap));



    network_to_caps(&cap);



    if (cap.version < 1 || cap.version > RDMA_CONTROL_VERSION_CURRENT) {

            fprintf(stderr, "Unknown source RDMA version: %d, bailing...\n",

                            cap.version);

            rdma_ack_cm_event(cm_event);

            goto err_rdma_dest_wait;

    }



    /*

     * Respond with only the capabilities this version of QEMU knows about.

     */

    cap.flags &= known_capabilities;



    /*

     * Enable the ones that we do know about.

     * Add other checks here as new ones are introduced.

     */

    if (cap.flags & RDMA_CAPABILITY_PIN_ALL) {

        rdma->pin_all = true;

    }



    rdma->cm_id = cm_event->id;

    verbs = cm_event->id->verbs;



    rdma_ack_cm_event(cm_event);



    DPRINTF("Memory pin all: %s\n", rdma->pin_all ? "enabled" : "disabled");



    caps_to_network(&cap);



    DPRINTF("verbs context after listen: %p\n", verbs);



    if (!rdma->verbs) {

        rdma->verbs = verbs;

    } else if (rdma->verbs != verbs) {

            fprintf(stderr, "ibv context not matching %p, %p!\n",

                    rdma->verbs, verbs);

            goto err_rdma_dest_wait;

    }



    qemu_rdma_dump_id("dest_init", verbs);



    ret = qemu_rdma_alloc_pd_cq(rdma);

    if (ret) {

        fprintf(stderr, "rdma migration: error allocating pd and cq!\n");

        goto err_rdma_dest_wait;

    }



    ret = qemu_rdma_alloc_qp(rdma);

    if (ret) {

        fprintf(stderr, "rdma migration: error allocating qp!\n");

        goto err_rdma_dest_wait;

    }



    ret = qemu_rdma_init_ram_blocks(rdma);

    if (ret) {

        fprintf(stderr, "rdma migration: error initializing ram blocks!\n");

        goto err_rdma_dest_wait;

    }



    for (idx = 0; idx < RDMA_WRID_MAX; idx++) {

        ret = qemu_rdma_reg_control(rdma, idx);

        if (ret) {

            fprintf(stderr, "rdma: error registering %d control!\n", idx);

            goto err_rdma_dest_wait;

        }

    }



    qemu_set_fd_handler2(rdma->channel->fd, NULL, NULL, NULL, NULL);



    ret = rdma_accept(rdma->cm_id, &conn_param);

    if (ret) {

        fprintf(stderr, "rdma_accept returns %d!\n", ret);

        goto err_rdma_dest_wait;

    }



    ret = rdma_get_cm_event(rdma->channel, &cm_event);

    if (ret) {

        fprintf(stderr, "rdma_accept get_cm_event failed %d!\n", ret);

        goto err_rdma_dest_wait;

    }



    if (cm_event->event != RDMA_CM_EVENT_ESTABLISHED) {

        fprintf(stderr, "rdma_accept not event established!\n");

        rdma_ack_cm_event(cm_event);

        goto err_rdma_dest_wait;

    }



    rdma_ack_cm_event(cm_event);

    rdma->connected = true;



    ret = qemu_rdma_post_recv_control(rdma, RDMA_WRID_READY);

    if (ret) {

        fprintf(stderr, "rdma migration: error posting second control recv!\n");

        goto err_rdma_dest_wait;

    }



    qemu_rdma_dump_gid("dest_connect", rdma->cm_id);



    return 0;



err_rdma_dest_wait:

    rdma->error_state = ret;

    qemu_rdma_cleanup(rdma);

    return ret;

}
