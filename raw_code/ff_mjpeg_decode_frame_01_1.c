int ff_mjpeg_decode_frame(AVCodecContext *avctx,

                              void *data, int *data_size,

                              AVPacket *avpkt)

{

    const uint8_t *buf = avpkt->data;

    int buf_size = avpkt->size;

    MJpegDecodeContext *s = avctx->priv_data;

    const uint8_t *buf_end, *buf_ptr;

    int start_code;

    AVFrame *picture = data;



    s->got_picture = 0; // picture from previous image can not be reused

    buf_ptr = buf;

    buf_end = buf + buf_size;

    while (buf_ptr < buf_end) {

        /* find start next marker */

        start_code = find_marker(&buf_ptr, buf_end);

        {

            /* EOF */

            if (start_code < 0) {

                goto the_end;

            } else {

                av_log(avctx, AV_LOG_DEBUG, "marker=%x avail_size_in_buf=%td\n", start_code, buf_end - buf_ptr);



                if ((buf_end - buf_ptr) > s->buffer_size)

                {

                    av_free(s->buffer);

                    s->buffer_size = buf_end-buf_ptr;

                    s->buffer = av_malloc(s->buffer_size + FF_INPUT_BUFFER_PADDING_SIZE);

                    av_log(avctx, AV_LOG_DEBUG, "buffer too small, expanding to %d bytes\n",

                        s->buffer_size);




                /* unescape buffer of SOS, use special treatment for JPEG-LS */

                if (start_code == SOS && !s->ls)

                {

                    const uint8_t *src = buf_ptr;

                    uint8_t *dst = s->buffer;



                    while (src<buf_end)

                    {

                        uint8_t x = *(src++);



                        *(dst++) = x;

                        if (avctx->codec_id != CODEC_ID_THP)

                        {

                            if (x == 0xff) {

                                while (src < buf_end && x == 0xff)

                                    x = *(src++);



                                if (x >= 0xd0 && x <= 0xd7)

                                    *(dst++) = x;

                                else if (x)





                    init_get_bits(&s->gb, s->buffer, (dst - s->buffer)*8);



                    av_log(avctx, AV_LOG_DEBUG, "escaping removed %td bytes\n",

                           (buf_end - buf_ptr) - (dst - s->buffer));


                else if(start_code == SOS && s->ls){

                    const uint8_t *src = buf_ptr;

                    uint8_t *dst = s->buffer;

                    int bit_count = 0;

                    int t = 0, b = 0;

                    PutBitContext pb;



                    s->cur_scan++;



                    /* find marker */

                    while (src + t < buf_end){

                        uint8_t x = src[t++];

                        if (x == 0xff){

                            while((src + t < buf_end) && x == 0xff)

                                x = src[t++];

                            if (x & 0x80) {

                                t -= 2;





                    bit_count = t * 8;



                    init_put_bits(&pb, dst, t);



                    /* unescape bitstream */

                    while(b < t){

                        uint8_t x = src[b++];

                        put_bits(&pb, 8, x);

                        if(x == 0xFF){

                            x = src[b++];

                            put_bits(&pb, 7, x);

                            bit_count--;



                    flush_put_bits(&pb);



                    init_get_bits(&s->gb, dst, bit_count);


                else

                    init_get_bits(&s->gb, buf_ptr, (buf_end - buf_ptr)*8);



                s->start_code = start_code;

                if(s->avctx->debug & FF_DEBUG_STARTCODE){

                    av_log(avctx, AV_LOG_DEBUG, "startcode: %X\n", start_code);




                /* process markers */

                if (start_code >= 0xd0 && start_code <= 0xd7) {

                    av_log(avctx, AV_LOG_DEBUG, "restart marker: %d\n", start_code&0x0f);

                    /* APP fields */

                } else if (start_code >= APP0 && start_code <= APP15) {

                    mjpeg_decode_app(s);

                    /* Comment */

                } else if (start_code == COM){

                    mjpeg_decode_com(s);




                switch(start_code) {

                case SOI:

                    s->restart_interval = 0;



                    s->restart_count = 0;

                    /* nothing to do on SOI */


                case DQT:

                    ff_mjpeg_decode_dqt(s);


                case DHT:

                    if(ff_mjpeg_decode_dht(s) < 0){

                        av_log(avctx, AV_LOG_ERROR, "huffman table decode error\n");

                        return -1;



                case SOF0:

                    s->lossless=0;

                    s->ls=0;

                    s->progressive=0;

                    if (ff_mjpeg_decode_sof(s) < 0)

                        return -1;


                case SOF2:

                    s->lossless=0;

                    s->ls=0;

                    s->progressive=1;

                    if (ff_mjpeg_decode_sof(s) < 0)

                        return -1;


                case SOF3:

                    s->lossless=1;

                    s->ls=0;

                    s->progressive=0;

                    if (ff_mjpeg_decode_sof(s) < 0)

                        return -1;


                case SOF48:

                    s->lossless=1;

                    s->ls=1;

                    s->progressive=0;

                    if (ff_mjpeg_decode_sof(s) < 0)

                        return -1;


                case LSE:

                    if (!CONFIG_JPEGLS_DECODER || ff_jpegls_decode_lse(s) < 0)

                        return -1;


                case EOI:

                    s->cur_scan = 0;

                    if ((s->buggy_avid && !s->interlaced) || s->restart_interval)


eoi_parser:


                        av_log(avctx, AV_LOG_WARNING, "Found EOI before any SOF, ignoring\n");



                    {

                        if (s->interlaced) {

                            s->bottom_field ^= 1;

                            /* if not bottom field, do not output image yet */

                            if (s->bottom_field == !s->interlace_polarity)

                                goto not_the_end;


                        *picture = s->picture;

                        *data_size = sizeof(AVFrame);



                        if(!s->lossless){

                            picture->quality= FFMAX3(s->qscale[0], s->qscale[1], s->qscale[2]);

                            picture->qstride= 0;

                            picture->qscale_table= s->qscale_table;

                            memset(picture->qscale_table, picture->quality, (s->width+15)/16);

                            if(avctx->debug & FF_DEBUG_QP)

                                av_log(avctx, AV_LOG_DEBUG, "QP: %d\n", picture->quality);

                            picture->quality*= FF_QP2LAMBDA;




                        goto the_end;



                case SOS:





                    ff_mjpeg_decode_sos(s);

                    /* buggy avid puts EOI every 10-20th frame */

                    /* if restart period is over process EOI */

                    if ((s->buggy_avid && !s->interlaced) || s->restart_interval)

                        goto eoi_parser;


                case DRI:

                    mjpeg_decode_dri(s);


                case SOF1:

                case SOF5:

                case SOF6:

                case SOF7:

                case SOF9:

                case SOF10:

                case SOF11:

                case SOF13:

                case SOF14:

                case SOF15:

                case JPG:

                    av_log(avctx, AV_LOG_ERROR, "mjpeg: unsupported coding type (%x)\n", start_code);


//                default:

//                    printf("mjpeg: unsupported marker (%x)\n", start_code);

//                    break;




not_the_end:

                /* eof process start code */

                buf_ptr += (get_bits_count(&s->gb)+7)/8;

                av_log(avctx, AV_LOG_DEBUG, "marker parser used %d bytes (%d bits)\n",

                       (get_bits_count(&s->gb)+7)/8, get_bits_count(&s->gb));




the_end:

    av_log(avctx, AV_LOG_DEBUG, "mjpeg decode frame unused %td bytes\n", buf_end - buf_ptr);

//    return buf_end - buf_ptr;

    return buf_ptr - buf;
