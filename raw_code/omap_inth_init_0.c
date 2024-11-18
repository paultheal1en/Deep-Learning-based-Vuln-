struct omap_intr_handler_s *omap_inth_init(target_phys_addr_t base,

                unsigned long size, unsigned char nbanks, qemu_irq **pins,

                qemu_irq parent_irq, qemu_irq parent_fiq, omap_clk clk)

{

    struct omap_intr_handler_s *s = (struct omap_intr_handler_s *)

            g_malloc0(sizeof(struct omap_intr_handler_s) +

                            sizeof(struct omap_intr_handler_bank_s) * nbanks);



    s->parent_intr[0] = parent_irq;

    s->parent_intr[1] = parent_fiq;

    s->nbanks = nbanks;

    s->pins = qemu_allocate_irqs(omap_set_intr, s, nbanks * 32);

    if (pins)

        *pins = s->pins;



    memory_region_init_io(&s->mmio, &omap_inth_mem_ops, s, "omap-intc", size);

    memory_region_add_subregion(get_system_memory(), base, &s->mmio);



    omap_inth_reset(s);



    return s;

}
