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




    s->sr[0x06] = 0x0f;

    if (device_id == CIRRUS_ID_CLGD5446) {

        /* 4MB 64 bit memory config, always PCI */

        s->sr[0x1F] = 0x2d;		// MemClock

        s->gr[0x18] = 0x0f;             // fastest memory configuration

#if 1

        s->sr[0x0f] = 0x98;

        s->sr[0x17] = 0x20;

        s->sr[0x15] = 0x04; /* memory size, 3=2MB, 4=4MB */

        s->real_vram_size = 4096 * 1024;

#else

        s->sr[0x0f] = 0x18;

        s->sr[0x17] = 0x20;

        s->sr[0x15] = 0x03; /* memory size, 3=2MB, 4=4MB */

        s->real_vram_size = 2048 * 1024;

#endif

    } else {

        s->sr[0x1F] = 0x22;		// MemClock

        s->sr[0x0F] = CIRRUS_MEMSIZE_2M;

        if (is_pci)

            s->sr[0x17] = CIRRUS_BUSTYPE_PCI;

        else

            s->sr[0x17] = CIRRUS_BUSTYPE_ISA;

        s->real_vram_size = 2048 * 1024;

        s->sr[0x15] = 0x03; /* memory size, 3=2MB, 4=4MB */

    }

    s->cr[0x27] = device_id;



    /* Win2K seems to assume that the pattern buffer is at 0xff

       initially ! */

    memset(s->vram_ptr, 0xff, s->real_vram_size);



    s->cirrus_hidden_dac_lockindex = 5;

    s->cirrus_hidden_dac_data = 0;



    /* I/O handler for LFB */

    s->cirrus_linear_io_addr =

	cpu_register_io_memory(0, cirrus_linear_read, cirrus_linear_write,

			       s);

    s->cirrus_linear_write = cpu_get_io_memory_write(s->cirrus_linear_io_addr);



    /* I/O handler for LFB */

    s->cirrus_linear_bitblt_io_addr =

	cpu_register_io_memory(0, cirrus_linear_bitblt_read, cirrus_linear_bitblt_write,

			       s);



    /* I/O handler for memory-mapped I/O */

    s->cirrus_mmio_io_addr =

	cpu_register_io_memory(0, cirrus_mmio_read, cirrus_mmio_write, s);



    /* XXX: s->vram_size must be a power of two */

    s->cirrus_addr_mask = s->real_vram_size - 1;

    s->linear_mmio_mask = s->real_vram_size - 256;



    s->get_bpp = cirrus_get_bpp;

    s->get_offsets = cirrus_get_offsets;

    s->get_resolution = cirrus_get_resolution;

    s->cursor_invalidate = cirrus_cursor_invalidate;

    s->cursor_draw_line = cirrus_cursor_draw_line;



    register_savevm("cirrus_vga", 0, 2, cirrus_vga_save, cirrus_vga_load, s);

}