static int mov_write_moov_tag(AVIOContext *pb, MOVMuxContext *mov,

                              AVFormatContext *s)

{

    int i;

    int64_t pos = avio_tell(pb);

    avio_wb32(pb, 0); /* size placeholder*/

    ffio_wfourcc(pb, "moov");



    for (i = 0; i < mov->nb_streams; i++) {

        if (mov->tracks[i].entry <= 0 && !(mov->flags & FF_MOV_FLAG_FRAGMENT))

            continue;



        mov->tracks[i].time     = mov->time;

        mov->tracks[i].track_id = i + 1;



        if (mov->tracks[i].entry)

            build_chunks(&mov->tracks[i]);

    }



    if (mov->chapter_track)

        for (i = 0; i < s->nb_streams; i++) {

            mov->tracks[i].tref_tag = MKTAG('c','h','a','p');

            mov->tracks[i].tref_id  = mov->tracks[mov->chapter_track].track_id;

        }

    for (i = 0; i < mov->nb_streams; i++) {

        if (mov->tracks[i].tag == MKTAG('r','t','p',' ')) {

            mov->tracks[i].tref_tag = MKTAG('h','i','n','t');

            mov->tracks[i].tref_id =

                mov->tracks[mov->tracks[i].src_track].track_id;

        }

    }

    for (i = 0; i < mov->nb_streams; i++) {

        if (mov->tracks[i].tag == MKTAG('t','m','c','d')) {

            int src_trk = mov->tracks[i].src_track;

            mov->tracks[src_trk].tref_tag = mov->tracks[i].tag;

            mov->tracks[src_trk].tref_id  = mov->tracks[i].track_id;

            //src_trk may have a different timescale than the tmcd track

            mov->tracks[i].track_duration = av_rescale(mov->tracks[src_trk].track_duration,

                                                       mov->tracks[i].timescale,

                                                       mov->tracks[src_trk].timescale);

        }

    }



    mov_write_mvhd_tag(pb, mov);

    if (mov->mode != MODE_MOV && !mov->iods_skip)

        mov_write_iods_tag(pb, mov);

    for (i = 0; i < mov->nb_streams; i++) {

        if (mov->tracks[i].entry > 0 || mov->flags & FF_MOV_FLAG_FRAGMENT) {

            mov_write_trak_tag(pb, mov, &(mov->tracks[i]), i < s->nb_streams ? s->streams[i] : NULL);

        }

    }

    if (mov->flags & FF_MOV_FLAG_FRAGMENT)

        mov_write_mvex_tag(pb, mov); /* QuickTime requires trak to precede this */



    if (mov->mode == MODE_PSP)

        mov_write_uuidusmt_tag(pb, s);

    else

        mov_write_udta_tag(pb, mov, s);



    return update_size(pb, pos);

}
