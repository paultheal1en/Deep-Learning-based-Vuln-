static void musicpal_lcd_write(void *opaque, target_phys_addr_t offset,

                               uint64_t value, unsigned size)

{

    musicpal_lcd_state *s = opaque;



    switch (offset) {

    case MP_LCD_IRQCTRL:

        s->irqctrl = value;

        break;



    case MP_LCD_SPICTRL:

        if (value == MP_LCD_SPI_DATA || value == MP_LCD_SPI_CMD) {

            s->mode = value;

        } else {

            s->mode = MP_LCD_SPI_INVALID;

        }

        break;



    case MP_LCD_INST:

        if (value >= MP_LCD_INST_SETPAGE0 && value <= MP_LCD_INST_SETPAGE7) {

            s->page = value - MP_LCD_INST_SETPAGE0;

            s->page_off = 0;

        }

        break;



    case MP_LCD_DATA:

        if (s->mode == MP_LCD_SPI_CMD) {

            if (value >= MP_LCD_INST_SETPAGE0 &&

                value <= MP_LCD_INST_SETPAGE7) {

                s->page = value - MP_LCD_INST_SETPAGE0;

                s->page_off = 0;

            }

        } else if (s->mode == MP_LCD_SPI_DATA) {

            s->video_ram[s->page*128 + s->page_off] = value;

            s->page_off = (s->page_off + 1) & 127;

        }

        break;

    }

}
