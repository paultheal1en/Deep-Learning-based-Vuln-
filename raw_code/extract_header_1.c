static int extract_header(AVCodecContext *const avctx,

                          const AVPacket *const avpkt) {

    const uint8_t *buf;

    unsigned buf_size;

    IffContext *s = avctx->priv_data;

    int palette_size;



    if (avctx->extradata_size < 2) {

        av_log(avctx, AV_LOG_ERROR, "not enough extradata\n");

        return AVERROR_INVALIDDATA;


    palette_size = avctx->extradata_size - AV_RB16(avctx->extradata);



    if (avpkt) {

        int image_size;

        if (avpkt->size < 2)

            return AVERROR_INVALIDDATA;

        image_size = avpkt->size - AV_RB16(avpkt->data);

        buf = avpkt->data;

        buf_size = bytestream_get_be16(&buf);

        if (buf_size <= 1 || image_size <= 1) {

            av_log(avctx, AV_LOG_ERROR,

                   "Invalid image size received: %u -> image data offset: %d\n",

                   buf_size, image_size);

            return AVERROR_INVALIDDATA;


    } else {

        buf = avctx->extradata;

        buf_size = bytestream_get_be16(&buf);

        if (buf_size <= 1 || palette_size < 0) {

            av_log(avctx, AV_LOG_ERROR,

                   "Invalid palette size received: %u -> palette data offset: %d\n",

                   buf_size, palette_size);

            return AVERROR_INVALIDDATA;





    if (buf_size > 8) {

        s->compression  = bytestream_get_byte(&buf);

        s->bpp          = bytestream_get_byte(&buf);

        s->ham          = bytestream_get_byte(&buf);

        s->flags        = bytestream_get_byte(&buf);

        s->transparency = bytestream_get_be16(&buf);

        s->masking      = bytestream_get_byte(&buf);

        if (s->masking == MASK_HAS_MASK) {

            if (s->bpp >= 8) {

                avctx->pix_fmt = PIX_FMT_RGB32;


                av_freep(&s->mask_palbuf);

                s->mask_buf = av_malloc((s->planesize * 32) + FF_INPUT_BUFFER_PADDING_SIZE);

                if (!s->mask_buf)







                s->mask_palbuf = av_malloc((2 << s->bpp) * sizeof(uint32_t) + FF_INPUT_BUFFER_PADDING_SIZE);

                if (!s->mask_palbuf) {





            s->bpp++;

        } else if (s->masking != MASK_NONE && s->masking != MASK_HAS_TRANSPARENT_COLOR) {

            av_log(avctx, AV_LOG_ERROR, "Masking not supported\n");

            return AVERROR_PATCHWELCOME;


        if (!s->bpp || s->bpp > 32) {

            av_log(avctx, AV_LOG_ERROR, "Invalid number of bitplanes: %u\n", s->bpp);

            return AVERROR_INVALIDDATA;

        } else if (s->ham >= 8) {

            av_log(avctx, AV_LOG_ERROR, "Invalid number of hold bits for HAM: %u\n", s->ham);

            return AVERROR_INVALIDDATA;




        av_freep(&s->ham_buf);

        av_freep(&s->ham_palbuf);



        if (s->ham) {

            int i, count = FFMIN(palette_size / 3, 1 << s->ham);

            int ham_count;

            const uint8_t *const palette = avctx->extradata + AV_RB16(avctx->extradata);



            s->ham_buf = av_malloc((s->planesize * 8) + FF_INPUT_BUFFER_PADDING_SIZE);

            if (!s->ham_buf)




            ham_count = 8 * (1 << s->ham);

            s->ham_palbuf = av_malloc((ham_count << !!(s->masking == MASK_HAS_MASK)) * sizeof (uint32_t) + FF_INPUT_BUFFER_PADDING_SIZE);

            if (!s->ham_palbuf) {

                av_freep(&s->ham_buf);





            if (count) { // HAM with color palette attached

                // prefill with black and palette and set HAM take direct value mask to zero

                memset(s->ham_palbuf, 0, (1 << s->ham) * 2 * sizeof (uint32_t));

                for (i=0; i < count; i++) {

                    s->ham_palbuf[i*2+1] = 0xFF000000 | AV_RL24(palette + i*3);


                count = 1 << s->ham;

            } else { // HAM with grayscale color palette

                count = 1 << s->ham;

                for (i=0; i < count; i++) {

                    s->ham_palbuf[i*2]   = 0xFF000000; // take direct color value from palette

                    s->ham_palbuf[i*2+1] = 0xFF000000 | av_le2ne32(gray2rgb((i * 255) >> s->ham));



            for (i=0; i < count; i++) {

                uint32_t tmp = i << (8 - s->ham);

                tmp |= tmp >> s->ham;

                s->ham_palbuf[(i+count)*2]     = 0xFF00FFFF; // just modify blue color component

                s->ham_palbuf[(i+count*2)*2]   = 0xFFFFFF00; // just modify red color component

                s->ham_palbuf[(i+count*3)*2]   = 0xFFFF00FF; // just modify green color component

                s->ham_palbuf[(i+count)*2+1]   = 0xFF000000 | tmp << 16;

                s->ham_palbuf[(i+count*2)*2+1] = 0xFF000000 | tmp;

                s->ham_palbuf[(i+count*3)*2+1] = 0xFF000000 | tmp << 8;


            if (s->masking == MASK_HAS_MASK) {

                for (i = 0; i < ham_count; i++)

                    s->ham_palbuf[(1 << s->bpp) + i] = s->ham_palbuf[i] | 0xFF000000;






    return 0;
