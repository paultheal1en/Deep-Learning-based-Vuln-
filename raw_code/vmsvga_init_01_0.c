static void vmsvga_init(struct vmsvga_state_s *s, DisplayState *ds,

                uint8_t *vga_ram_base, unsigned long vga_ram_offset,

                int vga_ram_size)

{

    s->ds = ds;

    s->vram = vga_ram_base;

    s->vram_size = vga_ram_size;

    s->vram_offset = vga_ram_offset;



    s->scratch_size = SVGA_SCRATCH_SIZE;

    s->scratch = (uint32_t *) qemu_malloc(s->scratch_size * 4);



    vmsvga_reset(s);



    s->console = graphic_console_init(ds, vmsvga_update_display,

                                      vmsvga_invalidate_display,

                                      vmsvga_screen_dump,

                                      vmsvga_text_update, s);



#ifdef EMBED_STDVGA

    vga_common_init((VGAState *) s, ds,

                    vga_ram_base, vga_ram_offset, vga_ram_size);

    vga_init((VGAState *) s);

#endif

}
