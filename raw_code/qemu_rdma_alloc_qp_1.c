static int qemu_rdma_alloc_qp(RDMAContext *rdma)

{

    struct ibv_qp_init_attr attr = { 0 };

    int ret;



    attr.cap.max_send_wr = RDMA_SIGNALED_SEND_MAX;

    attr.cap.max_recv_wr = 3;

    attr.cap.max_send_sge = 1;

    attr.cap.max_recv_sge = 1;

    attr.send_cq = rdma->cq;

    attr.recv_cq = rdma->cq;

    attr.qp_type = IBV_QPT_RC;



    ret = rdma_create_qp(rdma->cm_id, rdma->pd, &attr);

    if (ret) {

        return -1;

    }



    rdma->qp = rdma->cm_id->qp;

    return 0;

}
