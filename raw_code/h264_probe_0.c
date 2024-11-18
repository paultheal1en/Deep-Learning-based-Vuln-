static int h264_probe(AVProbeData *p)

{

    uint32_t code = -1;

    int sps = 0, pps = 0, idr = 0, res = 0, sli = 0;

    int i;



    for (i = 0; i < p->buf_size; i++) {

        code = (code << 8) + p->buf[i];

        if ((code & 0xffffff00) == 0x100) {

            int ref_idc = (code >> 5) & 3;

            int type    = code & 0x1F;

            static const int8_t ref_zero[] = {

                 2,  0,  0,  0,  0, -1,  1, -1,

                -1,  1,  1,  1,  1, -1,  2,  2,

                 2,  2,  2,  0,  2,  2,  2,  2,

                 2,  2,  2,  2,  2,  2,  2,  2

            };



            if (code & 0x80) // forbidden_bit

                return 0;



            if (ref_zero[type] == 1 && ref_idc)

                return 0;

            if (ref_zero[type] == -1 && !ref_idc)

                return 0;

            if (ref_zero[type] == 2) {

                if (!(code == 0x100 && !p->buf[i + 1] && !p->buf[i + 2]))

                    res++;

            }



            switch (type) {

            case 1:

                sli++;

                break;

            case 5:

                idr++;

                break;

            case 7:

                if (p->buf[i + 2] & 0x03)

                    return 0;

                sps++;

                break;

            case 8:

                pps++;

                break;

            }

        }

    }

    ff_tlog(NULL, "sps:%d pps:%d idr:%d sli:%d res:%d\n", sps, pps, idr, sli, res);



    if (sps && pps && (idr || sli > 3) && res < (sps + pps + idr))

        return AVPROBE_SCORE_EXTENSION + 1;  // 1 more than .mpg



    return 0;

}
