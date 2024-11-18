static uint32_t slavio_led_mem_reads(void *opaque, target_phys_addr_t addr)

{

    MiscState *s = opaque;

    uint32_t ret = 0, saddr;



    saddr = addr & LED_MAXADDR;

    switch (saddr) {

    case 0:

        ret = s->leds;

        break;

    default:

        break;

    }

    MISC_DPRINTF("Read diagnostic LED reg 0x" TARGET_FMT_plx " = %x\n", addr,

                 ret);

    return ret;

}
