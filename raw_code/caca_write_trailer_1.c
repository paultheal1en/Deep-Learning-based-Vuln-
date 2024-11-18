static int caca_write_trailer(AVFormatContext *s)

{

    CACAContext *c = s->priv_data;



    av_freep(&c->window_title);



    caca_free_dither(c->dither);

    caca_free_display(c->display);

    caca_free_canvas(c->canvas);

    return 0;

}
