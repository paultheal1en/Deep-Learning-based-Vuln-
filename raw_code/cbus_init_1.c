CBus *cbus_init(qemu_irq dat)

{

    CBusPriv *s = (CBusPriv *) g_malloc0(sizeof(*s));



    s->dat_out = dat;

    s->cbus.clk = qemu_allocate_irqs(cbus_clk, s, 1)[0];

    s->cbus.dat = qemu_allocate_irqs(cbus_dat, s, 1)[0];

    s->cbus.sel = qemu_allocate_irqs(cbus_sel, s, 1)[0];



    s->sel = 1;

    s->clk = 0;

    s->dat = 0;



    return &s->cbus;

}
