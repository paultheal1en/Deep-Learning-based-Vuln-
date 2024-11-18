static int decode_group3_1d_line(AVCodecContext *avctx, GetBitContext *gb,

                                 int pix_left, int *runs)

{

    int mode = 0, run = 0;

    unsigned int t;

    for(;;){

        t = get_vlc2(gb, ccitt_vlc[mode].table, 9, 2);

        run += t;

        if(t < 64){

            pix_left -= run;

            *runs++ = run;

            if(pix_left <= 0){

                if(!pix_left)

                    break;

                runs[-1] = 0;

                av_log(avctx, AV_LOG_ERROR, "Run went out of bounds\n");

                return -1;

            }

            run = 0;

            mode = !mode;

        }else if((int)t == -1){

            av_log(avctx, AV_LOG_ERROR, "Incorrect code\n");

            return -1;

        }

    }

    *runs++ = 0;

    return 0;

}
