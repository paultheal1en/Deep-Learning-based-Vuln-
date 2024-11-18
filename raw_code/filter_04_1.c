static void filter(MpegAudioContext *s, int ch, short *samples, int incr)

{

    short *p, *q;

    int sum, offset, i, j, norm, n;

    short tmp[64];

    int tmp1[32];

    int *out;



    //    print_pow1(samples, 1152);



    offset = s->samples_offset[ch];

    out = &s->sb_samples[ch][0][0][0];

    for(j=0;j<36;j++) {

        /* 32 samples at once */

        for(i=0;i<32;i++) {

            s->samples_buf[ch][offset + (31 - i)] = samples[0];

            samples += incr;

        }



        /* filter */

        p = s->samples_buf[ch] + offset;

        q = filter_bank;

        /* maxsum = 23169 */

        for(i=0;i<64;i++) {

            sum = p[0*64] * q[0*64];

            sum += p[1*64] * q[1*64];

            sum += p[2*64] * q[2*64];

            sum += p[3*64] * q[3*64];

            sum += p[4*64] * q[4*64];

            sum += p[5*64] * q[5*64];

            sum += p[6*64] * q[6*64];

            sum += p[7*64] * q[7*64];

            tmp[i] = sum >> 14;

            p++;

            q++;

        }

        tmp1[0] = tmp[16];

        for( i=1; i<=16; i++ ) tmp1[i] = tmp[i+16]+tmp[16-i];

        for( i=17; i<=31; i++ ) tmp1[i] = tmp[i+16]-tmp[80-i];



        /* integer IDCT 32 with normalization. XXX: There may be some

           overflow left */

        norm = 0;

        for(i=0;i<32;i++) {

            norm |= abs(tmp1[i]);

        }

        n = av_log2(norm) - 12;

        if (n > 0) {

            for(i=0;i<32;i++) 

                tmp1[i] >>= n;

        } else {

            n = 0;

        }



        idct32(out, tmp1, s->sblimit, n);



        /* advance of 32 samples */

        offset -= 32;

        out += 32;

        /* handle the wrap around */

        if (offset < 0) {

            memmove(s->samples_buf[ch] + SAMPLES_BUF_SIZE - (512 - 32), 

                    s->samples_buf[ch], (512 - 32) * 2);

            offset = SAMPLES_BUF_SIZE - 512;

        }

    }

    s->samples_offset[ch] = offset;



    //    print_pow(s->sb_samples, 1152);

}
