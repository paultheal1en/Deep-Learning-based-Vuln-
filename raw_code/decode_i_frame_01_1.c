static int decode_i_frame(FourXContext *f, AVFrame *frame, const uint8_t *buf, int length)

{

    int x, y, ret;

    const int width  = f->avctx->width;

    const int height = f->avctx->height;

    const unsigned int bitstream_size = AV_RL32(buf);

    int token_count av_unused;

    unsigned int prestream_size;

    const uint8_t *prestream;



    if (length < bitstream_size + 12) {

        av_log(f->avctx, AV_LOG_ERROR, "packet size too small\n");

        return AVERROR_INVALIDDATA;

    }



    token_count    =     AV_RL32(buf + bitstream_size + 8);

    prestream_size = 4 * AV_RL32(buf + bitstream_size + 4);

    prestream      =             buf + bitstream_size + 12;



    if (prestream_size + bitstream_size + 12 != length

        || bitstream_size > (1 << 26)

        || prestream_size > (1 << 26)) {

        av_log(f->avctx, AV_LOG_ERROR, "size mismatch %d %d %d\n",

               prestream_size, bitstream_size, length);

        return AVERROR_INVALIDDATA;

    }



    prestream = read_huffman_tables(f, prestream, prestream_size);

    if (!prestream) {

        av_log(f->avctx, AV_LOG_ERROR, "Error reading Huffman tables.\n");

        return AVERROR_INVALIDDATA;

    }



    init_get_bits(&f->gb, buf + 4, 8 * bitstream_size);



    prestream_size = length + buf - prestream;



    av_fast_malloc(&f->bitstream_buffer, &f->bitstream_buffer_size,

                   prestream_size + FF_INPUT_BUFFER_PADDING_SIZE);

    if (!f->bitstream_buffer)

        return AVERROR(ENOMEM);

    f->dsp.bswap_buf(f->bitstream_buffer, (const uint32_t*)prestream,

                     prestream_size / 4);

    memset((uint8_t*)f->bitstream_buffer + prestream_size,

           0, FF_INPUT_BUFFER_PADDING_SIZE);

    init_get_bits(&f->pre_gb, f->bitstream_buffer, 8 * prestream_size);



    f->last_dc = 0 * 128 * 8 * 8;



    for (y = 0; y < height; y += 16) {

        for (x = 0; x < width; x += 16) {

            if ((ret = decode_i_mb(f)) < 0)

                return ret;



            idct_put(f, frame, x, y);

        }

    }



    if (get_vlc2(&f->pre_gb, f->pre_vlc.table, ACDC_VLC_BITS, 3) != 256)

        av_log(f->avctx, AV_LOG_ERROR, "end mismatch\n");



    return 0;

}
