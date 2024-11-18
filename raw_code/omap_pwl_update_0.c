static void omap_pwl_update(struct omap_pwl_s *s)

{

    int output = (s->clk && s->enable) ? s->level : 0;



    if (output != s->output) {

        s->output = output;

        printf("%s: Backlight now at %i/256\n", __FUNCTION__, output);

    }

}
