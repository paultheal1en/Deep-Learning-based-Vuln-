static void vga_invalidate_display(void *opaque)

{

    VGAState *s = (VGAState *)opaque;



    s->last_width = -1;

    s->last_height = -1;

}
