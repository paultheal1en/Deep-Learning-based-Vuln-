static uint16_t nvme_io_cmd(NvmeCtrl *n, NvmeCmd *cmd, NvmeRequest *req)

{

    NvmeNamespace *ns;

    uint32_t nsid = le32_to_cpu(cmd->nsid);



    if (nsid == 0 || nsid > n->num_namespaces) {

        return NVME_INVALID_NSID | NVME_DNR;

    }



    ns = &n->namespaces[nsid - 1];

    switch (cmd->opcode) {

    case NVME_CMD_FLUSH:

        return nvme_flush(n, ns, cmd, req);

    case NVME_CMD_WRITE_ZEROS:

        return nvme_write_zeros(n, ns, cmd, req);

    case NVME_CMD_WRITE:

    case NVME_CMD_READ:

        return nvme_rw(n, ns, cmd, req);

    default:

        return NVME_INVALID_OPCODE | NVME_DNR;

    }

}
