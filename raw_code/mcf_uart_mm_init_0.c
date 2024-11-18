void mcf_uart_mm_init(MemoryRegion *sysmem,

                      target_phys_addr_t base,

                      qemu_irq irq,

                      CharDriverState *chr)

{

    mcf_uart_state *s;



    s = mcf_uart_init(irq, chr);

    memory_region_init_io(&s->iomem, &mcf_uart_ops, s, "uart", 0x40);

    memory_region_add_subregion(sysmem, base, &s->iomem);

}
