static void targa_decode_rle(AVCodecContext *avctx, TargaContext *s, const uint8_t *src, uint8_t *dst, int w, int h, int stride, int bpp)

{

    int i, x, y;

    int depth = (bpp + 1) >> 3;

    int type, count;

    int diff;



    diff = stride - w * depth;

    x = y = 0;

    while(y < h){

        type = *src++;

        count = (type & 0x7F) + 1;

        type &= 0x80;

        if((x + count > w) && (x + count + 1 > (h - y) * w)){

            av_log(avctx, AV_LOG_ERROR, "Packet went out of bounds: position (%i,%i) size %i\n", x, y, count);

            return;

        }

        for(i = 0; i < count; i++){

            switch(depth){

            case 1:

                *dst = *src;

                break;

            case 2:

                *((uint16_t*)dst) = AV_RL16(src);

                break;

            case 3:

                dst[0] = src[0];

                dst[1] = src[1];

                dst[2] = src[2];

                break;

            case 4:

                *((uint32_t*)dst) = AV_RL32(src);

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

}
