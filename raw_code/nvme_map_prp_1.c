static uint16_t nvme_map_prp(QEMUSGList *qsg, QEMUIOVector *iov, uint64_t prp1,

                             uint64_t prp2, uint32_t len, NvmeCtrl *n)

{

    hwaddr trans_len = n->page_size - (prp1 % n->page_size);

    trans_len = MIN(len, trans_len);

    int num_prps = (len >> n->page_bits) + 1;



    if (!prp1) {

        return NVME_INVALID_FIELD | NVME_DNR;

    } else if (n->cmbsz && prp1 >= n->ctrl_mem.addr &&

               prp1 < n->ctrl_mem.addr + int128_get64(n->ctrl_mem.size)) {

        qsg->nsg = 0;

        qemu_iovec_init(iov, num_prps);

        qemu_iovec_add(iov, (void *)&n->cmbuf[prp1 - n->ctrl_mem.addr], trans_len);

    } else {

        pci_dma_sglist_init(qsg, &n->parent_obj, num_prps);

        qemu_sglist_add(qsg, prp1, trans_len);

    }

    len -= trans_len;

    if (len) {

        if (!prp2) {

            goto unmap;

        }

        if (len > n->page_size) {

            uint64_t prp_list[n->max_prp_ents];

            uint32_t nents, prp_trans;

            int i = 0;



            nents = (len + n->page_size - 1) >> n->page_bits;

            prp_trans = MIN(n->max_prp_ents, nents) * sizeof(uint64_t);

            nvme_addr_read(n, prp2, (void *)prp_list, prp_trans);

            while (len != 0) {

                uint64_t prp_ent = le64_to_cpu(prp_list[i]);



                if (i == n->max_prp_ents - 1 && len > n->page_size) {

                    if (!prp_ent || prp_ent & (n->page_size - 1)) {

                        goto unmap;

                    }



                    i = 0;

                    nents = (len + n->page_size - 1) >> n->page_bits;

                    prp_trans = MIN(n->max_prp_ents, nents) * sizeof(uint64_t);

                    nvme_addr_read(n, prp_ent, (void *)prp_list,

                        prp_trans);

                    prp_ent = le64_to_cpu(prp_list[i]);

                }



                if (!prp_ent || prp_ent & (n->page_size - 1)) {

                    goto unmap;

                }



                trans_len = MIN(len, n->page_size);

                if (qsg->nsg){

                    qemu_sglist_add(qsg, prp_ent, trans_len);

                } else {

                    qemu_iovec_add(iov, (void *)&n->cmbuf[prp_ent - n->ctrl_mem.addr], trans_len);

                }

                len -= trans_len;

                i++;

            }

        } else {

            if (prp2 & (n->page_size - 1)) {

                goto unmap;

            }

            if (qsg->nsg) {

                qemu_sglist_add(qsg, prp2, len);

            } else {

                qemu_iovec_add(iov, (void *)&n->cmbuf[prp2 - n->ctrl_mem.addr], trans_len);

            }

        }

    }

    return NVME_SUCCESS;



 unmap:

    qemu_sglist_destroy(qsg);

    return NVME_INVALID_FIELD | NVME_DNR;

}
