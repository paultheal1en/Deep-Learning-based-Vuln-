static int mjpeg_probe(AVProbeData *p)

{

    int i;

    int state = -1;

    int nb_invalid = 0;

    int nb_frames = 0;



    for (i=0; i<p->buf_size-2; i++) {

        int c;

        if (p->buf[i] != 0xFF)

            continue;

        c = p->buf[i+1];

        switch (c) {

        case 0xD8:

            state = 0xD8;

            break;

        case 0xC0:

        case 0xC1:

        case 0xC2:

        case 0xC3:

        case 0xC5:

        case 0xC6:

        case 0xC7:

        case 0xF7:

            if (state == 0xD8) {

                state = 0xC0;

            } else

                nb_invalid++;

            break;

        case 0xDA:

            if (state == 0xC0) {

                state = 0xDA;

            } else

                nb_invalid++;

            break;

        case 0xD9:

            if (state == 0xDA) {

                state = 0xD9;

                nb_frames++;

            } else

                nb_invalid++;

            break;

        default:

            if (  (c >= 0x02 && c <= 0xBF)

                || c == 0xC8) {

                nb_invalid++;

            }

        }

    }



    if (nb_invalid*4 + 1 < nb_frames) {

        static const char ct_jpeg[] = "\r\nContent-Type: image/jpeg\r\n\r\n";

        int i;



        for (i=0; i<FFMIN(p->buf_size - sizeof(ct_jpeg), 100); i++)

            if (!memcmp(p->buf + i, ct_jpeg, sizeof(ct_jpeg) - 1))

                return AVPROBE_SCORE_EXTENSION;



        if (nb_invalid == 0 && nb_frames > 2)

            return AVPROBE_SCORE_EXTENSION / 2;

        return AVPROBE_SCORE_EXTENSION / 4;

    }



    return 0;

}
