static uint64_t gpio_read(void *opaque, target_phys_addr_t addr, unsigned size)

{

    struct gpio_state_t *s = opaque;

    uint32_t r = 0;



    addr >>= 2;

    switch (addr)

    {

        case R_PA_DIN:

            r = s->regs[RW_PA_DOUT] & s->regs[RW_PA_OE];



            /* Encode pins from the nand.  */

            r |= s->nand->rdy << 7;

            break;

        case R_PD_DIN:

            r = s->regs[RW_PD_DOUT] & s->regs[RW_PD_OE];



            /* Encode temp sensor pins.  */

            r |= (!!(s->tempsensor.shiftreg & 0x10000)) << 4;

            break;



        default:

            r = s->regs[addr];

            break;

    }

    return r;

    D(printf("%s %x=%x\n", __func__, addr, r));

}
