static void cirrus_linear_write(void *opaque, target_phys_addr_t addr,

                                uint64_t val, unsigned size)

{

    CirrusVGAState *s = opaque;

    unsigned mode;



    addr &= s->cirrus_addr_mask;



    if (((s->vga.sr[0x17] & 0x44) == 0x44) &&

        ((addr & s->linear_mmio_mask) ==  s->linear_mmio_mask)) {

	/* memory-mapped I/O */

	cirrus_mmio_blt_write(s, addr & 0xff, val);

    } else if (s->cirrus_srcptr != s->cirrus_srcptr_end) {

	/* bitblt */

	*s->cirrus_srcptr++ = (uint8_t) val;

	if (s->cirrus_srcptr >= s->cirrus_srcptr_end) {

	    cirrus_bitblt_cputovideo_next(s);

	}

    } else {

	/* video memory */

	if ((s->vga.gr[0x0B] & 0x14) == 0x14) {

	    addr <<= 4;

	} else if (s->vga.gr[0x0B] & 0x02) {

	    addr <<= 3;

	}

	addr &= s->cirrus_addr_mask;



	mode = s->vga.gr[0x05] & 0x7;

	if (mode < 4 || mode > 5 || ((s->vga.gr[0x0B] & 0x4) == 0)) {

	    *(s->vga.vram_ptr + addr) = (uint8_t) val;

            memory_region_set_dirty(&s->vga.vram, addr, 1);

	} else {

	    if ((s->vga.gr[0x0B] & 0x14) != 0x14) {

		cirrus_mem_writeb_mode4and5_8bpp(s, mode, addr, val);

	    } else {

		cirrus_mem_writeb_mode4and5_16bpp(s, mode, addr, val);

	    }

	}

    }

}
