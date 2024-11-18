static uint16_t nvme_identify_ns(NvmeCtrl *n, NvmeIdentify *c)

{

    NvmeNamespace *ns;

    uint32_t nsid = le32_to_cpu(c->nsid);

    uint64_t prp1 = le64_to_cpu(c->prp1);

    uint64_t prp2 = le64_to_cpu(c->prp2);



    if (nsid == 0 || nsid > n->num_namespaces) {

        return NVME_INVALID_NSID | NVME_DNR;

    }



    ns = &n->namespaces[nsid - 1];

    return nvme_dma_read_prp(n, (uint8_t *)&ns->id_ns, sizeof(ns->id_ns),

        prp1, prp2);

}
