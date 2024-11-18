static uint16_t nvme_create_sq(NvmeCtrl *n, NvmeCmd *cmd)

{

    NvmeSQueue *sq;

    NvmeCreateSq *c = (NvmeCreateSq *)cmd;



    uint16_t cqid = le16_to_cpu(c->cqid);

    uint16_t sqid = le16_to_cpu(c->sqid);

    uint16_t qsize = le16_to_cpu(c->qsize);

    uint16_t qflags = le16_to_cpu(c->sq_flags);

    uint64_t prp1 = le64_to_cpu(c->prp1);



    if (!cqid || nvme_check_cqid(n, cqid)) {

        return NVME_INVALID_CQID | NVME_DNR;

    }

    if (!sqid || !nvme_check_sqid(n, sqid)) {

        return NVME_INVALID_QID | NVME_DNR;

    }

    if (!qsize || qsize > NVME_CAP_MQES(n->bar.cap)) {

        return NVME_MAX_QSIZE_EXCEEDED | NVME_DNR;

    }

    if (!prp1 || prp1 & (n->page_size - 1)) {

        return NVME_INVALID_FIELD | NVME_DNR;

    }

    if (!(NVME_SQ_FLAGS_PC(qflags))) {

        return NVME_INVALID_FIELD | NVME_DNR;

    }

    sq = g_malloc0(sizeof(*sq));

    nvme_init_sq(sq, n, prp1, sqid, cqid, qsize + 1);

    return NVME_SUCCESS;

}
