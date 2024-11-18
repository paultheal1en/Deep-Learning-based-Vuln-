static int ra144_decode_frame(AVCodecContext * avctx,

            void *vdata, int *data_size,

            const uint8_t * buf, int buf_size)

{

    static const uint8_t sizes[10] = {6, 5, 5, 4, 4, 3, 3, 3, 3, 2};

    unsigned int a, b, c;

    int i;

    signed short *shptr;

    int16_t *data = vdata;

    unsigned int val;



    Real144_internal *glob = avctx->priv_data;

    GetBitContext gb;



    if(buf_size == 0)

        return 0;



    init_get_bits(&gb, buf, 20 * 8);



    for (i=0; i<10; i++)

        // "<< 1"? Doesn't this make one value out of two of the table useless?

        glob->swapbuf1[i] = decodetable[i][get_bits(&gb, sizes[i]) << 1];



    do_voice(glob->swapbuf1, glob->swapbuf2);



    val = decodeval[get_bits(&gb, 5) << 1]; // Useless table entries?

    a = t_sqrt(val*glob->oldval) >> 12;



    for (c=0; c < NBLOCKS; c++) {

        if (c == (NBLOCKS - 1)) {

            dec1(glob, glob->swapbuf1, glob->swapbuf2, 3, val);

        } else {

            if (c * 2 == (NBLOCKS - 2)) {

                if (glob->oldval < val) {

                    dec2(glob, glob->swapbuf1, glob->swapbuf2, 3, a, glob->swapbuf2alt, c);

                } else {

                    dec2(glob, glob->swapbuf1alt, glob->swapbuf2alt, 3, a, glob->swapbuf2, c);

                }

            } else {

                if (c * 2 < (NBLOCKS - 2)) {

                    dec2(glob, glob->swapbuf1alt, glob->swapbuf2alt, 3, glob->oldval, glob->swapbuf2, c);

                } else {

                    dec2(glob, glob->swapbuf1, glob->swapbuf2, 3, val, glob->swapbuf2alt, c);

                }

            }

        }

    }



    /* do output */

    for (b=0, c=0; c<4; c++) {

        unsigned int gval = glob->gbuf1[c * 2];

        unsigned short *gsp = glob->gbuf2 + b;

        signed short output_buffer[40];



        do_output_subblock(glob, gsp, gval, output_buffer, &gb);



        shptr = output_buffer;

        while (shptr < output_buffer + BLOCKSIZE)

            *data++ = av_clip_int16(*(shptr++) << 2);

        b += 30;

    }



    glob->oldval = val;



    FFSWAP(unsigned int *, glob->swapbuf1alt, glob->swapbuf1);

    FFSWAP(unsigned int *, glob->swapbuf2alt, glob->swapbuf2);



    *data_size = 2*160;

    return 20;

}
