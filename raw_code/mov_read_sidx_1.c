static int mov_read_sidx(MOVContext *c, AVIOContext *pb, MOVAtom atom)

{

    int64_t offset = avio_tell(pb) + atom.size, pts, timestamp;

    uint8_t version;

    unsigned i, j, track_id, item_count;

    AVStream *st = NULL;

    AVStream *ref_st = NULL;

    MOVStreamContext *sc, *ref_sc = NULL;

    AVRational timescale;



    version = avio_r8(pb);

    if (version > 1) {

        avpriv_request_sample(c->fc, "sidx version %u", version);

        return 0;

    }



    avio_rb24(pb); // flags



    track_id = avio_rb32(pb); // Reference ID

    for (i = 0; i < c->fc->nb_streams; i++) {

        if (c->fc->streams[i]->id == track_id) {

            st = c->fc->streams[i];

            break;

        }

    }

    if (!st) {

        av_log(c->fc, AV_LOG_WARNING, "could not find corresponding track id %d\n", track_id);

        return 0;

    }



    sc = st->priv_data;



    timescale = av_make_q(1, avio_rb32(pb));



    if (timescale.den <= 0) {

        av_log(c->fc, AV_LOG_ERROR, "Invalid sidx timescale 1/%d\n", timescale.den);

        return AVERROR_INVALIDDATA;

    }



    if (version == 0) {

        pts = avio_rb32(pb);

        offset += avio_rb32(pb);

    } else {

        pts = avio_rb64(pb);

        offset += avio_rb64(pb);

    }



    avio_rb16(pb); // reserved



    item_count = avio_rb16(pb);



    for (i = 0; i < item_count; i++) {

        int index;

        MOVFragmentStreamInfo * frag_stream_info;

        uint32_t size = avio_rb32(pb);

        uint32_t duration = avio_rb32(pb);

        if (size & 0x80000000) {

            avpriv_request_sample(c->fc, "sidx reference_type 1");

            return AVERROR_PATCHWELCOME;

        }

        avio_rb32(pb); // sap_flags

        timestamp = av_rescale_q(pts, st->time_base, timescale);



        index = update_frag_index(c, offset);

        frag_stream_info = get_frag_stream_info(&c->frag_index, index, track_id);

        if (frag_stream_info)

            frag_stream_info->sidx_pts = timestamp;



        offset += size;

        pts += duration;

    }



    st->duration = sc->track_end = pts;



    sc->has_sidx = 1;



    if (offset == avio_size(pb)) {

        // Find first entry in fragment index that came from an sidx.

        // This will pretty much always be the first entry.

        for (i = 0; i < c->frag_index.nb_items; i++) {

            MOVFragmentIndexItem * item = &c->frag_index.item[i];

            for (j = 0; ref_st == NULL && j < item->nb_stream_info; j++) {

                MOVFragmentStreamInfo * si;

                si = &item->stream_info[j];

                if (si->sidx_pts != AV_NOPTS_VALUE) {

                    ref_st = c->fc->streams[i];

                    ref_sc = ref_st->priv_data;

                    break;

                }

            }

        }

        for (i = 0; i < c->fc->nb_streams; i++) {

            st = c->fc->streams[i];

            sc = st->priv_data;

            if (!sc->has_sidx) {

                st->duration = sc->track_end = av_rescale(ref_st->duration, sc->time_scale, ref_sc->time_scale);

            }

        }



        c->frag_index.complete = 1;

    }



    return 0;

}
