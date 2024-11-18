static int ogg_get_length(AVFormatContext *s)

{

    struct ogg *ogg = s->priv_data;

    int i;

    int64_t size, end;

    int streams_left=0;



    if(!s->pb->seekable)

        return 0;



// already set

    if (s->duration != AV_NOPTS_VALUE)

        return 0;



    size = avio_size(s->pb);

    if(size < 0)

        return 0;

    end = size > MAX_PAGE_SIZE? size - MAX_PAGE_SIZE: 0;



    ogg_save (s);

    avio_seek (s->pb, end, SEEK_SET);



    while (!ogg_read_page (s, &i)){

        if (ogg->streams[i].granule != -1 && ogg->streams[i].granule != 0 &&

            ogg->streams[i].codec) {

            s->streams[i]->duration =

                ogg_gptopts (s, i, ogg->streams[i].granule, NULL);

            if (s->streams[i]->start_time != AV_NOPTS_VALUE){

                s->streams[i]->duration -= s->streams[i]->start_time;

                streams_left-= (ogg->streams[i].got_start==-1);

                ogg->streams[i].got_start= 1;

            }else if(!ogg->streams[i].got_start){

                ogg->streams[i].got_start= -1;

                streams_left++;

            }

        }

    }



    ogg_restore (s, 0);



    ogg_save (s);

    avio_seek (s->pb, s->data_offset, SEEK_SET);

    ogg_reset(s);


    while (!ogg_packet(s, &i, NULL, NULL, NULL)) {


        int64_t pts = ogg_calc_pts(s, i, NULL);

        if (pts != AV_NOPTS_VALUE && s->streams[i]->start_time == AV_NOPTS_VALUE && !ogg->streams[i].got_start){

            s->streams[i]->duration -= pts;

            ogg->streams[i].got_start= 1;

            streams_left--;

        }else if(s->streams[i]->start_time != AV_NOPTS_VALUE && !ogg->streams[i].got_start){

            ogg->streams[i].got_start= 1;

            streams_left--;

        }

        }

            if(streams_left<=0)

                break;

    }

    ogg_restore (s, 0);



    return 0;

}