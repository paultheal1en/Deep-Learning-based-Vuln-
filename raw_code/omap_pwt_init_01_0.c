static void omap_pwt_init(target_phys_addr_t base, struct omap_mpu_state_s *s,

                omap_clk clk)

{

    int iomemtype;



    s->pwt.base = base;

    s->pwt.clk = clk;

    omap_pwt_reset(s);



    iomemtype = cpu_register_io_memory(0, omap_pwt_readfn,

                    omap_pwt_writefn, s);

    cpu_register_physical_memory(s->pwt.base, 0x800, iomemtype);

}
