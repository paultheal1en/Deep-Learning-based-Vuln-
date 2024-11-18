static int mov_write_moov_tag(ByteIOContext *pb, MOVContext *mov)

{

    int pos, i;

    pos = url_ftell(pb);

    put_be32(pb, 0); /* size placeholder*/

    put_tag(pb, "moov");

    mov->timescale = globalTimescale;



    for (i=0; i<MAX_STREAMS; i++) {

        if(mov->tracks[i].entry <= 0) continue;



        if(mov->tracks[i].enc->codec_type == CODEC_TYPE_VIDEO) {

            mov->tracks[i].timescale = mov->tracks[i].enc->frame_rate;

            mov->tracks[i].sampleDuration = mov->tracks[i].enc->frame_rate_base;

        }

        else if(mov->tracks[i].enc->codec_type == CODEC_TYPE_AUDIO) {

            /* If AMR, track timescale = 8000, AMR_WB = 16000 */

            if(mov->tracks[i].enc->codec_id == CODEC_ID_AMR_NB) {

                mov->tracks[i].sampleDuration = 160;  // Bytes per chunk

                mov->tracks[i].timescale = 8000;

            }

            else {

                mov->tracks[i].timescale = mov->tracks[i].enc->sample_rate;

                mov->tracks[i].sampleDuration = mov->tracks[i].enc->frame_size;

            }

        }



        mov->tracks[i].trackDuration = 

            mov->tracks[i].sampleCount * mov->tracks[i].sampleDuration;

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



    return updateSize(pb, pos);

}
