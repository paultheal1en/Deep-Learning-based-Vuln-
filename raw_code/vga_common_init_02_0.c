void vga_common_init(VGACommonState *s, int vga_ram_size)

{

    int i, j, v, b;



    for(i = 0;i < 256; i++) {

        v = 0;

        for(j = 0; j < 8; j++) {

            v |= ((i >> j) & 1) << (j * 4);

        }

        expand4[i] = v;



        v = 0;

        for(j = 0; j < 4; j++) {

            v |= ((i >> (2 * j)) & 3) << (j * 4);

        }

        expand2[i] = v;

    }

    for(i = 0; i < 16; i++) {

        v = 0;

        for(j = 0; j < 4; j++) {

            b = ((i >> j) & 1);

            v |= b << (2 * j);

            v |= b << (2 * j + 1);

        }

        expand4to8[i] = v;

    }



#ifdef CONFIG_BOCHS_VBE

    s->is_vbe_vmstate = 1;

#else

    s->is_vbe_vmstate = 0;

#endif

    s->vram_offset = qemu_ram_alloc(vga_ram_size);

    s->vram_ptr = qemu_get_ram_ptr(s->vram_offset);

    s->vram_size = vga_ram_size;

    s->get_bpp = vga_get_bpp;

    s->get_offsets = vga_get_offsets;

    s->get_resolution = vga_get_resolution;

    s->update = vga_update_display;

    s->invalidate = vga_invalidate_display;

    s->screen_dump = vga_screen_dump;

    s->text_update = vga_update_text;

    switch (vga_retrace_method) {

    case VGA_RETRACE_DUMB:

        s->retrace = vga_dumb_retrace;

        s->update_retrace_info = vga_dumb_update_retrace_info;

        break;



    case VGA_RETRACE_PRECISE:

        s->retrace = vga_precise_retrace;

        s->update_retrace_info = vga_precise_update_retrace_info;

        break;

    }

    vga_reset(s);

}
