static void vmsvga_text_update(void *opaque, console_ch_t *chardata)

{

    struct vmsvga_state_s *s = opaque;



    if (s->vga.text_update)

        s->vga.text_update(&s->vga, chardata);

}
