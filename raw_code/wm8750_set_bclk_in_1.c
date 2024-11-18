void wm8750_set_bclk_in(void *opaque, int hz)

{

    struct wm8750_s *s = (struct wm8750_s *) opaque;



    s->ext_adc_hz = hz;

    s->ext_dac_hz = hz;

    wm8750_clk_update(s, 1);

}
