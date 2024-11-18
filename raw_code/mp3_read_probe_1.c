static int mp3_read_probe(AVProbeData *p)

{

    int max_frames, first_frames;

    int fsize, frames, sample_rate;

    uint32_t header;

    uint8_t *buf, *buf2, *end;

    AVCodecContext avctx;



    if(id3v2_match(p->buf))

        return AVPROBE_SCORE_MAX/2+1; // this must be less than mpeg-ps because some retards put id3v2 tags before mpeg-ps files



    max_frames = 0;

    buf = p->buf;

    end = buf + FFMIN(4096, p->buf_size - sizeof(uint32_t));



    for(; buf < end; buf++) {

        buf2 = buf;



        for(frames = 0; buf2 < end; frames++) {

            header = AV_RB32(buf2);

            fsize = ff_mpa_decode_header(&avctx, header, &sample_rate);

            if(fsize < 0)

                break;

            buf2 += fsize;

        }

        max_frames = FFMAX(max_frames, frames);

        if(buf == p->buf)

            first_frames= frames;

    }

    if   (first_frames>=3) return AVPROBE_SCORE_MAX/2+1;

    else if(max_frames>=3) return AVPROBE_SCORE_MAX/4;

    else if(max_frames>=1) return 1;

    else                   return 0;

}
