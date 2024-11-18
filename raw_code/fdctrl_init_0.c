fdctrl_t *fdctrl_init (qemu_irq irq, int dma_chann, int mem_mapped,

                       target_phys_addr_t io_base,

                       BlockDriverState **fds)

{

    fdctrl_t *fdctrl;

    int io_mem;

    int i;



    FLOPPY_DPRINTF("init controller\n");

    fdctrl = qemu_mallocz(sizeof(fdctrl_t));

    if (!fdctrl)

        return NULL;

    fdctrl->fifo = qemu_memalign(512, FD_SECTOR_LEN);

    if (fdctrl->fifo == NULL) {

        qemu_free(fdctrl);

        return NULL;

    }

    fdctrl->result_timer = qemu_new_timer(vm_clock,

                                          fdctrl_result_timer, fdctrl);



    fdctrl->version = 0x90; /* Intel 82078 controller */

    fdctrl->irq = irq;

    fdctrl->dma_chann = dma_chann;

    fdctrl->io_base = io_base;

    fdctrl->config = 0x60; /* Implicit seek, polling & FIFO enabled */

    fdctrl->sun4m = 0;

    if (fdctrl->dma_chann != -1) {

        fdctrl->dma_en = 1;

        DMA_register_channel(dma_chann, &fdctrl_transfer_handler, fdctrl);

    } else {

        fdctrl->dma_en = 0;

    }

    for (i = 0; i < 2; i++) {

        fd_init(&fdctrl->drives[i], fds[i]);

    }

    fdctrl_reset(fdctrl, 0);

    fdctrl->state = FD_CTRL_ACTIVE;

    if (mem_mapped) {

        io_mem = cpu_register_io_memory(0, fdctrl_mem_read, fdctrl_mem_write,

                                        fdctrl);

        cpu_register_physical_memory(io_base, 0x08, io_mem);

    } else {

        register_ioport_read((uint32_t)io_base + 0x01, 5, 1, &fdctrl_read,

                             fdctrl);

        register_ioport_read((uint32_t)io_base + 0x07, 1, 1, &fdctrl_read,

                             fdctrl);

        register_ioport_write((uint32_t)io_base + 0x01, 5, 1, &fdctrl_write,

                              fdctrl);

        register_ioport_write((uint32_t)io_base + 0x07, 1, 1, &fdctrl_write,

                              fdctrl);

    }

    register_savevm("fdc", io_base, 1, fdc_save, fdc_load, fdctrl);

    qemu_register_reset(fdctrl_external_reset, fdctrl);

    for (i = 0; i < 2; i++) {

        fd_revalidate(&fdctrl->drives[i]);

    }



    return fdctrl;

}
