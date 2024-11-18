static void mm_decode_intra(MmContext * s, int half_horiz, int half_vert, const uint8_t *buf, int buf_size)
{
    int i, x, y;
    i=0; x=0; y=0;
    while(i<buf_size) {
        int run_length, color;
        if (buf[i] & 0x80) {
            run_length = 1;
            color = buf[i];
            i++;
        }else{
            run_length = (buf[i] & 0x7f) + 2;
            color = buf[i+1];
            i+=2;
        }
        if (half_horiz)
            run_length *=2;
        if (color) {
            memset(s->frame.data[0] + y*s->frame.linesize[0] + x, color, run_length);
            if (half_vert)
                memset(s->frame.data[0] + (y+1)*s->frame.linesize[0] + x, color, run_length);
        }
        x+= run_length;
        if (x >= s->avctx->width) {
            x=0;
            y += 1 + half_vert;
        }
    }
}