static int qemu_rdma_alloc_pd_cq(RDMAContext *rdma)

{

    /* allocate pd */

    rdma->pd = ibv_alloc_pd(rdma->verbs);

    if (!rdma->pd) {

        fprintf(stderr, "failed to allocate protection domain\n");

        return -1;

    }



    /* create completion channel */

    rdma->comp_channel = ibv_create_comp_channel(rdma->verbs);

    if (!rdma->comp_channel) {

        fprintf(stderr, "failed to allocate completion channel\n");

        goto err_alloc_pd_cq;

    }



    /*

     * Completion queue can be filled by both read and write work requests,

     * so must reflect the sum of both possible queue sizes.

     */

    rdma->cq = ibv_create_cq(rdma->verbs, (RDMA_SIGNALED_SEND_MAX * 3),

            NULL, rdma->comp_channel, 0);

    if (!rdma->cq) {

        fprintf(stderr, "failed to allocate completion queue\n");

        goto err_alloc_pd_cq;

    }



    return 0;



err_alloc_pd_cq:

    if (rdma->pd) {

        ibv_dealloc_pd(rdma->pd);

    }

    if (rdma->comp_channel) {

        ibv_destroy_comp_channel(rdma->comp_channel);

    }

    rdma->pd = NULL;

    rdma->comp_channel = NULL;

    return -1;



}
