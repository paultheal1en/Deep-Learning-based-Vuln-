static void slavio_led_mem_writes(void *opaque, target_phys_addr_t addr,

                                  uint32_t val)

{

    MiscState *s = opaque;

    uint32_t saddr;



    saddr = addr & LED_MAXADDR;

    MISC_DPRINTF("Write diagnostic LED reg 0x" TARGET_FMT_plx " =  %x\n", addr,

                 val);

    switch (saddr) {

    case 0:

        s->leds = val;

        break;

    default:

        break;

    }

}
