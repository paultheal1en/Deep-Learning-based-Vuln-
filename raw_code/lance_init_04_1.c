void lance_init(NICInfo *nd, target_phys_addr_t leaddr, void *dma_opaque,

                qemu_irq irq, qemu_irq *reset)

{

    PCNetState *d;

    int lance_io_memory;



    qemu_check_nic_model(nd, "lance");



    d = qemu_mallocz(sizeof(PCNetState));



    lance_io_memory =

        cpu_register_io_memory(0, lance_mem_read, lance_mem_write, d);



    d->dma_opaque = dma_opaque;



    *reset = *qemu_allocate_irqs(parent_lance_reset, d, 1);



    cpu_register_physical_memory(leaddr, 4, lance_io_memory);



    d->irq = irq;

    d->phys_mem_read = ledma_memory_read;

    d->phys_mem_write = ledma_memory_write;



    pcnet_common_init(d, nd);

}
