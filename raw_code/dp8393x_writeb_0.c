static void dp8393x_writeb(void *opaque, target_phys_addr_t addr, uint32_t val)

{

    uint16_t old_val = dp8393x_readw(opaque, addr & ~0x1);



    switch (addr & 3) {

    case 0:

        val = val | (old_val & 0xff00);

        break;

    case 1:

        val = (val << 8) | (old_val & 0x00ff);

        break;

    }

    dp8393x_writew(opaque, addr & ~0x1, val);

}
