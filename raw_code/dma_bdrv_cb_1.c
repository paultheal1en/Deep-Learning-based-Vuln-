static void dma_bdrv_cb(void *opaque, int ret)

{

    DMAAIOCB *dbs = (DMAAIOCB *)opaque;

    target_phys_addr_t cur_addr, cur_len;

    void *mem;



    dbs->acb = NULL;

    dbs->sector_num += dbs->iov.size / 512;

    dma_bdrv_unmap(dbs);

    qemu_iovec_reset(&dbs->iov);



    if (dbs->sg_cur_index == dbs->sg->nsg || ret < 0) {

        dbs->common.cb(dbs->common.opaque, ret);

        qemu_iovec_destroy(&dbs->iov);

        qemu_aio_release(dbs);

        return;

    }



    while (dbs->sg_cur_index < dbs->sg->nsg) {

        cur_addr = dbs->sg->sg[dbs->sg_cur_index].base + dbs->sg_cur_byte;

        cur_len = dbs->sg->sg[dbs->sg_cur_index].len - dbs->sg_cur_byte;

        mem = cpu_physical_memory_map(cur_addr, &cur_len, !dbs->to_dev);

        if (!mem)

            break;

        qemu_iovec_add(&dbs->iov, mem, cur_len);

        dbs->sg_cur_byte += cur_len;

        if (dbs->sg_cur_byte == dbs->sg->sg[dbs->sg_cur_index].len) {

            dbs->sg_cur_byte = 0;

            ++dbs->sg_cur_index;

        }

    }



    if (dbs->iov.size == 0) {

        cpu_register_map_client(dbs, continue_after_map_failure);

        return;

    }



    dbs->acb = dbs->io_func(dbs->bs, dbs->sector_num, &dbs->iov,

                            dbs->iov.size / 512, dma_bdrv_cb, dbs);

    if (!dbs->acb) {

        dma_bdrv_unmap(dbs);

        qemu_iovec_destroy(&dbs->iov);

        return;

    }

}
