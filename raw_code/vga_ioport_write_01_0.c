static void vga_ioport_write(void *opaque, uint32_t addr, uint32_t val)

{

    CirrusVGAState *s = opaque;

    int index;



    /* check port range access depending on color/monochrome mode */

    if ((addr >= 0x3b0 && addr <= 0x3bf && (s->msr & MSR_COLOR_EMULATION))

	|| (addr >= 0x3d0 && addr <= 0x3df

	    && !(s->msr & MSR_COLOR_EMULATION)))

	return;



#ifdef DEBUG_VGA

    printf("VGA: write addr=0x%04x data=0x%02x\n", addr, val);

#endif



    switch (addr) {

    case 0x3c0:

	if (s->ar_flip_flop == 0) {

	    val &= 0x3f;

	    s->ar_index = val;

	} else {

	    index = s->ar_index & 0x1f;

	    switch (index) {

	    case 0x00 ... 0x0f:

		s->ar[index] = val & 0x3f;

		break;

	    case 0x10:

		s->ar[index] = val & ~0x10;

		break;

	    case 0x11:

		s->ar[index] = val;

		break;

	    case 0x12:

		s->ar[index] = val & ~0xc0;

		break;

	    case 0x13:

		s->ar[index] = val & ~0xf0;

		break;

	    case 0x14:

		s->ar[index] = val & ~0xf0;

		break;

	    default:

		break;

	    }

	}

	s->ar_flip_flop ^= 1;

	break;

    case 0x3c2:

	s->msr = val & ~0x10;

	break;

    case 0x3c4:

	s->sr_index = val;

	break;

    case 0x3c5:

	if (cirrus_hook_write_sr(s, s->sr_index, val))

	    break;

#ifdef DEBUG_VGA_REG

	printf("vga: write SR%x = 0x%02x\n", s->sr_index, val);

#endif

	s->sr[s->sr_index] = val & sr_mask[s->sr_index];

	break;

    case 0x3c6:

	cirrus_write_hidden_dac(s, val);

	break;

    case 0x3c7:

	s->dac_read_index = val;

	s->dac_sub_index = 0;

	s->dac_state = 3;

	break;

    case 0x3c8:

	s->dac_write_index = val;

	s->dac_sub_index = 0;

	s->dac_state = 0;

	break;

    case 0x3c9:

	if (cirrus_hook_write_palette(s, val))

	    break;

	s->dac_cache[s->dac_sub_index] = val;

	if (++s->dac_sub_index == 3) {

	    memcpy(&s->palette[s->dac_write_index * 3], s->dac_cache, 3);

	    s->dac_sub_index = 0;

	    s->dac_write_index++;

	}

	break;

    case 0x3ce:

	s->gr_index = val;

	break;

    case 0x3cf:

	if (cirrus_hook_write_gr(s, s->gr_index, val))

	    break;

#ifdef DEBUG_VGA_REG

	printf("vga: write GR%x = 0x%02x\n", s->gr_index, val);

#endif

	s->gr[s->gr_index] = val & gr_mask[s->gr_index];

	break;

    case 0x3b4:

    case 0x3d4:

	s->cr_index = val;

	break;

    case 0x3b5:

    case 0x3d5:

	if (cirrus_hook_write_cr(s, s->cr_index, val))

	    break;

#ifdef DEBUG_VGA_REG

	printf("vga: write CR%x = 0x%02x\n", s->cr_index, val);

#endif

	/* handle CR0-7 protection */

	if ((s->cr[11] & 0x80) && s->cr_index <= 7) {

	    /* can always write bit 4 of CR7 */

	    if (s->cr_index == 7)

		s->cr[7] = (s->cr[7] & ~0x10) | (val & 0x10);

	    return;

	}

	switch (s->cr_index) {

	case 0x01:		/* horizontal display end */

	case 0x07:

	case 0x09:

	case 0x0c:

	case 0x0d:

	case 0x12:		/* veritcal display end */

	    s->cr[s->cr_index] = val;

	    break;



	default:

	    s->cr[s->cr_index] = val;

	    break;

	}

	break;

    case 0x3ba:

    case 0x3da:

	s->fcr = val & 0x10;

	break;

    }

}
