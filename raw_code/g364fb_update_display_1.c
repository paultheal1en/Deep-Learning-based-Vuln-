static void g364fb_update_display(void *opaque)
{
    G364State *s = opaque;
    if (s->width == 0 || s->height == 0)
        return;
    if (s->width != ds_get_width(s->ds) || s->height != ds_get_height(s->ds)) {
        qemu_console_resize(s->ds, s->width, s->height);
    }
    if (s->ctla & CTLA_FORCE_BLANK) {
        g364fb_draw_blank(s);
    } else if (s->depth == 8) {
        g364fb_draw_graphic8(s);
    } else {
        error_report("g364: unknown guest depth %d", s->depth);
    }
    qemu_irq_raise(s->irq);
}