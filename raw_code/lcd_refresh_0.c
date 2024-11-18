static void lcd_refresh(void *opaque)

{

    musicpal_lcd_state *s = opaque;

    int x, y;



    for (x = 0; x < 128; x++)

        for (y = 0; y < 64; y++)

            if (s->video_ram[x + (y/8)*128] & (1 << (y % 8)))

                set_lcd_pixel(s, x, y, MP_LCD_TEXTCOLOR);

            else

                set_lcd_pixel(s, x, y, 0);



    dpy_update(s->ds, 0, 0, 128*3, 64*3);

}
