static uint16_t nvme_get_feature(NvmeCtrl *n, NvmeCmd *cmd, NvmeRequest *req)

{

    uint32_t dw10 = le32_to_cpu(cmd->cdw10);

    uint32_t result;



    switch (dw10) {

    case NVME_VOLATILE_WRITE_CACHE:

        result = blk_enable_write_cache(n->conf.blk);

        break;

    case NVME_NUMBER_OF_QUEUES:

        result = cpu_to_le32((n->num_queues - 1) | ((n->num_queues - 1) << 16));

        break;

    default:

        return NVME_INVALID_FIELD | NVME_DNR;

    }



    req->cqe.result = result;

    return NVME_SUCCESS;

}
