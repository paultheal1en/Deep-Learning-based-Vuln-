void omap_mpuio_out_set(struct omap_mpuio_s *s, int line, qemu_irq handler)

{

    if (line >= 16 || line < 0)

        hw_error("%s: No GPIO line %i\n", __FUNCTION__, line);

    s->handler[line] = handler;

}
