static int mkv_write_trailer(AVFormatContext *s)

{

    MatroskaMuxContext *mkv = s->priv_data;

    AVIOContext *pb = s->pb;

    int64_t currentpos, cuespos;

    int ret;



    // check if we have an audio packet cached

    if (mkv->cur_audio_pkt.size > 0) {

        ret = mkv_write_packet_internal(s, &mkv->cur_audio_pkt);

        mkv->cur_audio_pkt.size = 0;

        if (ret < 0) {

            av_log(s, AV_LOG_ERROR, "Could not write cached audio packet ret:%d\n", ret);

            return ret;

        }

    }



    if (mkv->dyn_bc) {

        end_ebml_master(mkv->dyn_bc, mkv->cluster);

        mkv_flush_dynbuf(s);

    } else if (mkv->cluster_pos) {

        end_ebml_master(pb, mkv->cluster);

    }



    if (pb->seekable) {

        if (mkv->cues->num_entries) {

            cuespos = mkv_write_cues(pb, mkv->cues, s->nb_streams);



            ret = mkv_add_seekhead_entry(mkv->main_seekhead, MATROSKA_ID_CUES, cuespos);

            if (ret < 0) return ret;

        }



        mkv_write_seekhead(pb, mkv->main_seekhead);



        // update the duration

        av_log(s, AV_LOG_DEBUG, "end duration = %" PRIu64 "\n", mkv->duration);

        currentpos = avio_tell(pb);

        avio_seek(pb, mkv->duration_offset, SEEK_SET);

        put_ebml_float(pb, MATROSKA_ID_DURATION, mkv->duration);



        avio_seek(pb, currentpos, SEEK_SET);

    }



    end_ebml_master(pb, mkv->segment);

    av_free(mkv->tracks);

    av_freep(&mkv->cues->entries);

    av_freep(&mkv->cues);

    av_destruct_packet(&mkv->cur_audio_pkt);

    avio_flush(pb);

    return 0;

}
