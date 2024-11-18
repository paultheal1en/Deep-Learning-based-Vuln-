static int qemu_rdma_connect(RDMAContext *rdma, Error **errp)

{

    RDMACapabilities cap = {

                                .version = RDMA_CONTROL_VERSION_CURRENT,

                                .flags = 0,

                           };

    struct rdma_conn_param conn_param = { .initiator_depth = 2,

                                          .retry_count = 5,

                                          .private_data = &cap,

                                          .private_data_len = sizeof(cap),

                                        };

    struct rdma_cm_event *cm_event;

    int ret;



    /*

     * Only negotiate the capability with destination if the user

     * on the source first requested the capability.

     */

    if (rdma->pin_all) {

        trace_qemu_rdma_connect_pin_all_requested();

        cap.flags |= RDMA_CAPABILITY_PIN_ALL;

    }



    caps_to_network(&cap);



    ret = rdma_connect(rdma->cm_id, &conn_param);

    if (ret) {

        perror("rdma_connect");

        ERROR(errp, "connecting to destination!");

        goto err_rdma_source_connect;

    }



    ret = rdma_get_cm_event(rdma->channel, &cm_event);

    if (ret) {

        perror("rdma_get_cm_event after rdma_connect");

        ERROR(errp, "connecting to destination!");

        rdma_ack_cm_event(cm_event);

        goto err_rdma_source_connect;

    }



    if (cm_event->event != RDMA_CM_EVENT_ESTABLISHED) {

        perror("rdma_get_cm_event != EVENT_ESTABLISHED after rdma_connect");

        ERROR(errp, "connecting to destination!");

        rdma_ack_cm_event(cm_event);

        goto err_rdma_source_connect;

    }

    rdma->connected = true;



    memcpy(&cap, cm_event->param.conn.private_data, sizeof(cap));

    network_to_caps(&cap);



    /*

     * Verify that the *requested* capabilities are supported by the destination

     * and disable them otherwise.

     */

    if (rdma->pin_all && !(cap.flags & RDMA_CAPABILITY_PIN_ALL)) {

        ERROR(errp, "Server cannot support pinning all memory. "

                        "Will register memory dynamically.");

        rdma->pin_all = false;

    }



    trace_qemu_rdma_connect_pin_all_outcome(rdma->pin_all);



    rdma_ack_cm_event(cm_event);



    ret = qemu_rdma_post_recv_control(rdma, RDMA_WRID_READY);

    if (ret) {

        ERROR(errp, "posting second control recv!");

        goto err_rdma_source_connect;

    }



    rdma->control_ready_expected = 1;

    rdma->nb_sent = 0;

    return 0;



err_rdma_source_connect:

    qemu_rdma_cleanup(rdma);

    return -1;

}
