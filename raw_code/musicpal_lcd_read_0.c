static uint64_t musicpal_lcd_read(void *opaque, target_phys_addr_t offset,

                                  unsigned size)

{

    musicpal_lcd_state *s = opaque;



    switch (offset) {

    case MP_LCD_IRQCTRL:

        return s->irqctrl;



    default:

        return 0;

    }

}
