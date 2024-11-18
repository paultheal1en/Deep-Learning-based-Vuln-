static uint64_t musicpal_gpio_read(void *opaque, target_phys_addr_t offset,

                                   unsigned size)

{

    musicpal_gpio_state *s = opaque;



    switch (offset) {

    case MP_GPIO_OE_HI: /* used for LCD brightness control */

        return s->lcd_brightness & MP_OE_LCD_BRIGHTNESS;



    case MP_GPIO_OUT_LO:

        return s->out_state & 0xFFFF;

    case MP_GPIO_OUT_HI:

        return s->out_state >> 16;



    case MP_GPIO_IN_LO:

        return s->in_state & 0xFFFF;

    case MP_GPIO_IN_HI:

        return s->in_state >> 16;



    case MP_GPIO_IER_LO:

        return s->ier & 0xFFFF;

    case MP_GPIO_IER_HI:

        return s->ier >> 16;



    case MP_GPIO_IMR_LO:

        return s->imr & 0xFFFF;

    case MP_GPIO_IMR_HI:

        return s->imr >> 16;



    case MP_GPIO_ISR_LO:

        return s->isr & 0xFFFF;

    case MP_GPIO_ISR_HI:

        return s->isr >> 16;



    default:

        return 0;

    }

}
