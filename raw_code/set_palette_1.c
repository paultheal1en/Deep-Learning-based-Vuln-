static int set_palette(AVFrame * frame, const uint8_t * palette_buffer, int buf_size)

{

    uint32_t * palette = (uint32_t *)frame->data[1];

    int a;



    if (buf_size < 256*3)

        return AVERROR_INVALIDDATA;



    for(a = 0; a < 256; a++){

        palette[a] = AV_RB24(&palette_buffer[a * 3]) * 4;

    }

    frame->palette_has_changed = 1;

    return 256*3;

}
