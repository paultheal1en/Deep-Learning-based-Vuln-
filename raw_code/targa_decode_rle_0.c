static int targa_decode_rle(AVCodecContext *avctx, TargaContext *s, const uint8_t *src, int src_size, uint8_t *dst, int w, int h, int stride, int bpp)

{

    int i, x, y;

    int depth = (bpp + 1) >> 3;

    int type, count;

    int diff;

    const uint8_t *src_end = src + src_size;



    diff = stride - w * depth;

    x = y = 0;

    while(y < h){

        CHECK_BUFFER_SIZE(src, src_end, 1, "image type");

        type = *src++;

        count = (type & 0x7F) + 1;

        type &= 0x80;

        if((x + count > w) && (x + count + 1 > (h - y) * w)){

            av_log(avctx, AV_LOG_ERROR, "Packet went out of bounds: position (%i,%i) size %i\n", x, y, count);

            return -1;

        }

        if(type){

            CHECK_BUFFER_SIZE(src, src_end, depth, "image data");

        }else{

            CHECK_BUFFER_SIZE(src, src_end, count * depth, "image data");

        }

        for(i = 0; i < count; i++){

            switch(depth){

            case 1:

                *dst = *src;

                break;

            case 2:

                AV_WN16A(dst, AV_RN16A(src));

                break;

            case 3:

                dst[0] = src[0];

                dst[1] = src[1];

                dst[2] = src[2];

                break;

            case 4:

                AV_WN32A(dst, AV_RN32A(src));

                break;

            }

            dst += depth;

            if(!type)

                src += depth;



            x++;

            if(x == w){

                x = 0;

                y++;

                dst += diff;

            }

        }

        if(type)

            src += depth;

    }

    return src_size;

}
