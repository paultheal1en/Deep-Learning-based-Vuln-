static uint32_t omap_sysctl_read8(void *opaque, target_phys_addr_t addr)

{



    struct omap_sysctl_s *s = (struct omap_sysctl_s *) opaque;

    int pad_offset, byte_offset;

    int value;



    switch (addr) {

    case 0x030 ... 0x140:	/* CONTROL_PADCONF - only used in the POP */

        pad_offset = (addr - 0x30) >> 2;

        byte_offset = (addr - 0x30) & (4 - 1);



        value = s->padconf[pad_offset];

        value = (value >> (byte_offset * 8)) & 0xff;



        return value;



    default:

        break;

    }



    OMAP_BAD_REG(addr);

    return 0;

}
