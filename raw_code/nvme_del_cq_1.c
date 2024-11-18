static uint16_t nvme_del_cq(NvmeCtrl *n, NvmeCmd *cmd)

{

    NvmeDeleteQ *c = (NvmeDeleteQ *)cmd;

    NvmeCQueue *cq;

    uint16_t qid = le16_to_cpu(c->qid);



    if (!qid || nvme_check_cqid(n, qid)) {

        return NVME_INVALID_CQID | NVME_DNR;

    }



    cq = n->cq[qid];

    if (!QTAILQ_EMPTY(&cq->sq_list)) {

        return NVME_INVALID_QUEUE_DEL;

    }

    nvme_free_cq(cq, n);

    return NVME_SUCCESS;

}
