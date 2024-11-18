static void vmsvga_init(struct vmsvga_state_s *s,

                        MemoryRegion *address_space, MemoryRegion *io)

{

    DisplaySurface *surface;



    s->scratch_size = SVGA_SCRATCH_SIZE;

    s->scratch = g_malloc(s->scratch_size * 4);



    s->vga.con = graphic_console_init(vmsvga_update_display,

                                      vmsvga_invalidate_display,

                                      vmsvga_screen_dump,

                                      vmsvga_text_update, s);

    surface = qemu_console_surface(s->vga.con);



    s->fifo_size = SVGA_FIFO_SIZE;

    memory_region_init_ram(&s->fifo_ram, "vmsvga.fifo", s->fifo_size);

    vmstate_register_ram_global(&s->fifo_ram);

    s->fifo_ptr = memory_region_get_ram_ptr(&s->fifo_ram);



    vga_common_init(&s->vga);

    vga_init(&s->vga, address_space, io, true);

    vmstate_register(NULL, 0, &vmstate_vga_common, &s->vga);

    /* Save some values here in case they are changed later.

     * This is suspicious and needs more though why it is needed. */

    s->depth = surface_bits_per_pixel(surface);

    s->bypp = surface_bytes_per_pixel(surface);

}
