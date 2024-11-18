static int mpegps_probe(AVProbeData *p)

{

    uint32_t code= -1;

    int sys=0, pspack=0, priv1=0, vid=0, audio=0, invalid=0;

    int i;

    int score=0;



    for(i=0; i<p->buf_size; i++){

        code = (code<<8) + p->buf[i];

        if ((code & 0xffffff00) == 0x100) {

            int len= p->buf[i+1] << 8 | p->buf[i+2];

            int pes= check_pes(p->buf+i, p->buf+p->buf_size);



            if(code == SYSTEM_HEADER_START_CODE) sys++;

            else if(code == PACK_START_CODE)     pspack++;

            else if((code & 0xf0) == VIDEO_ID &&  pes) vid++;

            // skip pes payload to avoid start code emulation for private

            // and audio streams

            else if((code & 0xe0) == AUDIO_ID &&  pes) {audio++; i+=len;}

            else if(code == PRIVATE_STREAM_1  &&  pes) {priv1++; i+=len;}



            else if((code & 0xf0) == VIDEO_ID && !pes) invalid++;

            else if((code & 0xe0) == AUDIO_ID && !pes) invalid++;

            else if(code == PRIVATE_STREAM_1  && !pes) invalid++;

        }

    }



    if(vid+audio > invalid)     /* invalid VDR files nd short PES streams */

        score= AVPROBE_SCORE_MAX/4;



//av_log(NULL, AV_LOG_ERROR, "%d %d %d %d %d %d len:%d\n", sys, priv1, pspack,vid, audio, invalid, p->buf_size);

    if(sys>invalid && sys*9 <= pspack*10)

        return pspack > 2 ? AVPROBE_SCORE_MAX/2+2 : AVPROBE_SCORE_MAX/4; // +1 for .mpg

    if(pspack > invalid && (priv1+vid+audio)*10 >= pspack*9)

        return pspack > 2 ? AVPROBE_SCORE_MAX/2+2 : AVPROBE_SCORE_MAX/4; // +1 for .mpg

    if((!!vid ^ !!audio) && (audio > 4 || vid > 1) && !sys && !pspack && p->buf_size>2048 && vid + audio > invalid) /* PES stream */

        return (audio > 12 || vid > 3) ? AVPROBE_SCORE_MAX/2+2 : AVPROBE_SCORE_MAX/4;



    //02-Penguin.flac has sys:0 priv1:0 pspack:0 vid:0 audio:1

    //mp3_misidentified_2.mp3 has sys:0 priv1:0 pspack:0 vid:0 audio:6

    return score;

}
