static void tc6393xb_gpio_set(void *opaque, int line, int level)

{

//    TC6393xbState *s = opaque;



    if (line > TC6393XB_GPIOS) {

        printf("%s: No GPIO pin %i\n", __FUNCTION__, line);

        return;

    }



    // FIXME: how does the chip reflect the GPIO input level change?

}
