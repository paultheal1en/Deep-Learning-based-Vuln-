static int mov_write_moov_tag(ByteIOContext *pb, MOVContext *mov,

                              AVFormatContext *s)

{

    int i;

    offset_t pos = url_ftell(pb);

    put_be32(pb, 0); /* size placeholder*/

    put_tag(pb, "moov");

    mov->timescale = globalTimescale;



    for (i=0; i<MAX_STREAMS; i++) {

        if(mov->tracks[i].entry <= 0) continue;



        if(mov->tracks[i].enc->codec_type == CODEC_TYPE_VIDEO) {

            mov->tracks[i].timescale = mov->tracks[i].enc->time_base.den;

            mov->tracks[i].sampleDuration = mov->tracks[i].enc->time_base.num;

        } else if(mov->tracks[i].enc->codec_type == CODEC_TYPE_AUDIO) {

            mov->tracks[i].timescale = mov->tracks[i].enc->sample_rate;

            mov->tracks[i].sampleDuration = mov->tracks[i].enc->frame_size;

        }



        mov->tracks[i].trackDuration =

            (int64_t)mov->tracks[i].sampleCount * mov->tracks[i].sampleDuration;

        mov->tracks[i].time = mov->time;

        mov->tracks[i].trackID = i+1;

    }



    mov_write_mvhd_tag(pb, mov);

    //mov_write_iods_tag(pb, mov);

    for (i=0; i<MAX_STREAMS; i++) {

        if(mov->tracks[i].entry > 0) {

            mov_write_trak_tag(pb, &(mov->tracks[i]));

        }

    }



    if (mov->mode == MODE_PSP)

        mov_write_uuidusmt_tag(pb, s);

    else

    mov_write_udta_tag(pb, mov, s);



    return updateSize(pb, pos);

}
