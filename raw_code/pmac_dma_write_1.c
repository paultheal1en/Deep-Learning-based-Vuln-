static void pmac_dma_write(BlockBackend *blk,

                         int64_t sector_num, int nb_sectors,

                         void (*cb)(void *opaque, int ret), void *opaque)

{

    DBDMA_io *io = opaque;

    MACIOIDEState *m = io->opaque;

    IDEState *s = idebus_active_if(&m->bus);

    dma_addr_t dma_addr, dma_len;

    void *mem;

    int nsector, remainder;

    int extra = 0;



    qemu_iovec_destroy(&io->iov);

    qemu_iovec_init(&io->iov, io->len / MACIO_PAGE_SIZE + 1);



    if (io->remainder_len > 0) {

        /* Return remainder of request */

        int transfer = MIN(io->remainder_len, io->len);



        MACIO_DPRINTF("--- processing write remainder %x\n", transfer);

        cpu_physical_memory_read(io->addr,

                                 &io->remainder + (0x200 - transfer),

                                 transfer);



        io->remainder_len -= transfer;

        io->len -= transfer;

        io->addr += transfer;



        s->io_buffer_index += transfer;

        s->io_buffer_size -= transfer;



        if (io->remainder_len != 0) {

            /* Still waiting for remainder */

            return;

        }



        MACIO_DPRINTF("--> prepending bounce buffer with size 0x200\n");



        /* Sector transfer complete - prepend to request */

        qemu_iovec_add(&io->iov, &io->remainder, 0x200);

        extra = 1;

    }



    if (s->drive_kind == IDE_CD) {

        sector_num = (int64_t)(s->lba << 2) + (s->io_buffer_index >> 9);

    } else {

        sector_num = ide_get_sector(s) + (s->io_buffer_index >> 9);

    }



    nsector = (io->len >> 9);

    remainder = io->len - (nsector << 9);



    MACIO_DPRINTF("--- DMA write transfer - addr: %" HWADDR_PRIx " len: %x\n",

                  io->addr, io->len);

    MACIO_DPRINTF("xxx remainder: %x\n", remainder);

    MACIO_DPRINTF("xxx sector_num: %"PRIx64"   nsector: %x\n",

                  sector_num, nsector);



    dma_addr = io->addr;

    dma_len = io->len;

    mem = dma_memory_map(&address_space_memory, dma_addr, &dma_len,

                         DMA_DIRECTION_TO_DEVICE);



    if (!remainder) {

        MACIO_DPRINTF("--- DMA write aligned - addr: %" HWADDR_PRIx

                      " len: %x\n", io->addr, io->len);

        qemu_iovec_add(&io->iov, mem, io->len);

    } else {

        /* Write up to last complete sector */

        MACIO_DPRINTF("--- DMA write unaligned - addr: %" HWADDR_PRIx

                      " len: %x\n", io->addr, (nsector << 9));

        qemu_iovec_add(&io->iov, mem, (nsector << 9));



        MACIO_DPRINTF("--- DMA write read    - bounce addr: %p "

                      "remainder_len: %x\n", &io->remainder, remainder);

        cpu_physical_memory_read(io->addr + (nsector << 9), &io->remainder,

                                 remainder);



        io->remainder_len = 0x200 - remainder;



        MACIO_DPRINTF("xxx remainder_len: %x\n", io->remainder_len);

    }



    s->io_buffer_size -= ((nsector + extra) << 9);

    s->io_buffer_index += ((nsector + extra) << 9);



    io->len = 0;



    MACIO_DPRINTF("--- Block write transfer   - sector_num: %"PRIx64"  "

                  "nsector: %x\n", sector_num, nsector + extra);



    m->aiocb = blk_aio_writev(blk, sector_num, &io->iov, nsector + extra, cb,

                              io);

}
