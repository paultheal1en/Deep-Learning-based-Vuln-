static int nvme_start_ctrl(NvmeCtrl *n)

{

    uint32_t page_bits = NVME_CC_MPS(n->bar.cc) + 12;

    uint32_t page_size = 1 << page_bits;



    if (n->cq[0] || n->sq[0] || !n->bar.asq || !n->bar.acq ||

            n->bar.asq & (page_size - 1) || n->bar.acq & (page_size - 1) ||

            NVME_CC_MPS(n->bar.cc) < NVME_CAP_MPSMIN(n->bar.cap) ||

            NVME_CC_MPS(n->bar.cc) > NVME_CAP_MPSMAX(n->bar.cap) ||

            NVME_CC_IOCQES(n->bar.cc) < NVME_CTRL_CQES_MIN(n->id_ctrl.cqes) ||

            NVME_CC_IOCQES(n->bar.cc) > NVME_CTRL_CQES_MAX(n->id_ctrl.cqes) ||

            NVME_CC_IOSQES(n->bar.cc) < NVME_CTRL_SQES_MIN(n->id_ctrl.sqes) ||

            NVME_CC_IOSQES(n->bar.cc) > NVME_CTRL_SQES_MAX(n->id_ctrl.sqes) ||

            !NVME_AQA_ASQS(n->bar.aqa) || !NVME_AQA_ACQS(n->bar.aqa)) {

        return -1;

    }



    n->page_bits = page_bits;

    n->page_size = page_size;

    n->max_prp_ents = n->page_size / sizeof(uint64_t);

    n->cqe_size = 1 << NVME_CC_IOCQES(n->bar.cc);

    n->sqe_size = 1 << NVME_CC_IOSQES(n->bar.cc);

    nvme_init_cq(&n->admin_cq, n, n->bar.acq, 0, 0,

        NVME_AQA_ACQS(n->bar.aqa) + 1, 1);

    nvme_init_sq(&n->admin_sq, n, n->bar.asq, 0, 0,

        NVME_AQA_ASQS(n->bar.aqa) + 1);



    return 0;

}
