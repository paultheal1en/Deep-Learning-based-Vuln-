static void vb_decode_palette(VBDecContext *c)

{

    int start, size, i;



    start = bytestream_get_byte(&c->stream);

    size = (bytestream_get_byte(&c->stream) - 1) & 0xFF;

    if(start + size > 255){

        av_log(c->avctx, AV_LOG_ERROR, "Palette change runs beyond entry 256\n");

        return;

    }

    for(i = start; i <= start + size; i++)

        c->pal[i] = bytestream_get_be24(&c->stream);

}
