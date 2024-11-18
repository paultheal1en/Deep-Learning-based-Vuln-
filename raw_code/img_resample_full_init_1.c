ImgReSampleContext *img_resample_full_init(int owidth, int oheight,
                                      int iwidth, int iheight,
                                      int topBand, int bottomBand,
        int leftBand, int rightBand,
        int padtop, int padbottom,
        int padleft, int padright)
{
    ImgReSampleContext *s;
    s = av_mallocz(sizeof(ImgReSampleContext));
    if (!s)
    if((unsigned)owidth >= UINT_MAX / (LINE_BUF_HEIGHT + NB_TAPS))
    s->line_buf = av_mallocz(owidth * (LINE_BUF_HEIGHT + NB_TAPS));
    if (!s->line_buf) 
        goto fail;
    s->owidth = owidth;
    s->oheight = oheight;
    s->iwidth = iwidth;
    s->iheight = iheight;
    s->topBand = topBand;
    s->bottomBand = bottomBand;
    s->leftBand = leftBand;
    s->rightBand = rightBand;
    s->padtop = padtop;
    s->padbottom = padbottom;
    s->padleft = padleft;
    s->padright = padright;
    s->pad_owidth = owidth - (padleft + padright);
    s->pad_oheight = oheight - (padtop + padbottom);
    s->h_incr = ((iwidth - leftBand - rightBand) * POS_FRAC) / s->pad_owidth;
    s->v_incr = ((iheight - topBand - bottomBand) * POS_FRAC) / s->pad_oheight; 
    av_build_filter(&s->h_filters[0][0], (float) s->pad_owidth  / 
            (float) (iwidth - leftBand - rightBand), NB_TAPS, NB_PHASES, 1<<FILTER_BITS, 0);
    av_build_filter(&s->v_filters[0][0], (float) s->pad_oheight / 
            (float) (iheight - topBand - bottomBand), NB_TAPS, NB_PHASES, 1<<FILTER_BITS, 0);
    return s;
fail:
    av_free(s);
}