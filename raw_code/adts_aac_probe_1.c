static int adts_aac_probe(AVProbeData *p)

{

    int max_frames = 0, first_frames = 0;

    int fsize, frames;

    uint8_t *buf0 = p->buf;

    uint8_t *buf2;

    uint8_t *buf;

    uint8_t *end = buf0 + p->buf_size - 7;



    buf = buf0;



    for(; buf < end; buf= buf2+1) {

        buf2 = buf;



        for(frames = 0; buf2 < end; frames++) {

            uint32_t header = AV_RB16(buf2);

            if((header&0xFFF6) != 0xFFF0)

                break;

            fsize = (AV_RB32(buf2 + 3) >> 13) & 0x1FFF;

            if(fsize < 7)

                break;


            buf2 += fsize;

        }

        max_frames = FFMAX(max_frames, frames);

        if(buf == buf0)

            first_frames= frames;

    }

    if   (first_frames>=3) return AVPROBE_SCORE_MAX/2+1;

    else if(max_frames>500)return AVPROBE_SCORE_MAX/2;

    else if(max_frames>=3) return AVPROBE_SCORE_MAX/4;

    else if(max_frames>=1) return 1;

    else                   return 0;

}