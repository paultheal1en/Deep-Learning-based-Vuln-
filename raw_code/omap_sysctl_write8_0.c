static void omap_sysctl_write8(void *opaque, target_phys_addr_t addr,

                uint32_t value)

{

    struct omap_sysctl_s *s = (struct omap_sysctl_s *) opaque;

    int pad_offset, byte_offset;

    int prev_value;



    switch (addr) {

    case 0x030 ... 0x140:	/* CONTROL_PADCONF - only used in the POP */

        pad_offset = (addr - 0x30) >> 2;

        byte_offset = (addr - 0x30) & (4 - 1);



        prev_value = s->padconf[pad_offset];

        prev_value &= ~(0xff << (byte_offset * 8));

        prev_value |= ((value & 0x1f1f1f1f) << (byte_offset * 8)) & 0x1f1f1f1f;

        s->padconf[pad_offset] = prev_value;

        break;



    default:

        OMAP_BAD_REG(addr);

        break;

    }

}
