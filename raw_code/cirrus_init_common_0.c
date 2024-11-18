static void cirrus_init_common(CirrusVGAState * s, int device_id, int is_pci)

{

    int i;

    static int inited;



    if (!inited) {

        inited = 1;

        for(i = 0;i < 256; i++)

            rop_to_index[i] = CIRRUS_ROP_NOP_INDEX; /* nop rop */

        rop_to_index[CIRRUS_ROP_0] = 0;

        rop_to_index[CIRRUS_ROP_SRC_AND_DST] = 1;

        rop_to_index[CIRRUS_ROP_NOP] = 2;

        rop_to_index[CIRRUS_ROP_SRC_AND_NOTDST] = 3;

        rop_to_index[CIRRUS_ROP_NOTDST] = 4;

        rop_to_index[CIRRUS_ROP_SRC] = 5;

        rop_to_index[CIRRUS_ROP_1] = 6;

        rop_to_index[CIRRUS_ROP_NOTSRC_AND_DST] = 7;

        rop_to_index[CIRRUS_ROP_SRC_XOR_DST] = 8;

        rop_to_index[CIRRUS_ROP_SRC_OR_DST] = 9;

        rop_to_index[CIRRUS_ROP_NOTSRC_OR_NOTDST] = 10;

        rop_to_index[CIRRUS_ROP_SRC_NOTXOR_DST] = 11;

        rop_to_index[CIRRUS_ROP_SRC_OR_NOTDST] = 12;

        rop_to_index[CIRRUS_ROP_NOTSRC] = 13;

        rop_to_index[CIRRUS_ROP_NOTSRC_OR_DST] = 14;

        rop_to_index[CIRRUS_ROP_NOTSRC_AND_NOTDST] = 15;

        s->device_id = device_id;

        if (is_pci)

            s->bustype = CIRRUS_BUSTYPE_PCI;

        else

            s->bustype = CIRRUS_BUSTYPE_ISA;

    }



    register_ioport_write(0x3c0, 16, 1, vga_ioport_write, s);



    register_ioport_write(0x3b4, 2, 1, vga_ioport_write, s);

    register_ioport_write(0x3d4, 2, 1, vga_ioport_write, s);

    register_ioport_write(0x3ba, 1, 1, vga_ioport_write, s);

    register_ioport_write(0x3da, 1, 1, vga_ioport_write, s);



    register_ioport_read(0x3c0, 16, 1, vga_ioport_read, s);



    register_ioport_read(0x3b4, 2, 1, vga_ioport_read, s);

    register_ioport_read(0x3d4, 2, 1, vga_ioport_read, s);

    register_ioport_read(0x3ba, 1, 1, vga_ioport_read, s);

    register_ioport_read(0x3da, 1, 1, vga_ioport_read, s);



    s->vga_io_memory = cpu_register_io_memory(0, cirrus_vga_mem_read,

                                           cirrus_vga_mem_write, s);

    cpu_register_physical_memory(isa_mem_base + 0x000a0000, 0x20000,

                                 s->vga_io_memory);

    qemu_register_coalesced_mmio(isa_mem_base + 0x000a0000, 0x20000);



    /* I/O handler for LFB */

    s->cirrus_linear_io_addr =

        cpu_register_io_memory(0, cirrus_linear_read, cirrus_linear_write, s);

    s->cirrus_linear_write = cpu_get_io_memory_write(s->cirrus_linear_io_addr);



    /* I/O handler for LFB */

    s->cirrus_linear_bitblt_io_addr =

        cpu_register_io_memory(0, cirrus_linear_bitblt_read,

                               cirrus_linear_bitblt_write, s);



    /* I/O handler for memory-mapped I/O */

    s->cirrus_mmio_io_addr =

        cpu_register_io_memory(0, cirrus_mmio_read, cirrus_mmio_write, s);



    s->real_vram_size =

        (s->device_id == CIRRUS_ID_CLGD5446) ? 4096 * 1024 : 2048 * 1024;



    /* XXX: s->vram_size must be a power of two */

    s->cirrus_addr_mask = s->real_vram_size - 1;

    s->linear_mmio_mask = s->real_vram_size - 256;



    s->get_bpp = cirrus_get_bpp;

    s->get_offsets = cirrus_get_offsets;

    s->get_resolution = cirrus_get_resolution;

    s->cursor_invalidate = cirrus_cursor_invalidate;

    s->cursor_draw_line = cirrus_cursor_draw_line;



    qemu_register_reset(cirrus_reset, s);

    cirrus_reset(s);

    register_savevm("cirrus_vga", 0, 2, cirrus_vga_save, cirrus_vga_load, s);

}
