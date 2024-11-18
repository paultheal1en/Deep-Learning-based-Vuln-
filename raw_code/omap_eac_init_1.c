static struct omap_eac_s *omap_eac_init(struct omap_target_agent_s *ta,

                qemu_irq irq, qemu_irq *drq, omap_clk fclk, omap_clk iclk)

{

    struct omap_eac_s *s = (struct omap_eac_s *)

            g_malloc0(sizeof(struct omap_eac_s));



    s->irq = irq;

    s->codec.rxdrq = *drq ++;

    s->codec.txdrq = *drq;

    omap_eac_reset(s);



    AUD_register_card("OMAP EAC", &s->codec.card);



    memory_region_init_io(&s->iomem, NULL, &omap_eac_ops, s, "omap.eac",

                          omap_l4_region_size(ta, 0));

    omap_l4_attach(ta, 0, &s->iomem);



    return s;

}
