static int64_t mkv_write_cues(AVFormatContext *s, mkv_cues *cues, mkv_track *tracks, int num_tracks)

{

    MatroskaMuxContext *mkv = s->priv_data;

    AVIOContext *dyn_cp, *pb = s->pb;

    ebml_master cues_element;

    int64_t currentpos;

    int i, j, ret;



    currentpos = avio_tell(pb);

    ret = start_ebml_master_crc32(pb, &dyn_cp, &cues_element, MATROSKA_ID_CUES, 0);

    if (ret < 0)

        return ret;



    for (i = 0; i < cues->num_entries; i++) {

        ebml_master cuepoint, track_positions;

        mkv_cuepoint *entry = &cues->entries[i];

        uint64_t pts = entry->pts;

        int ctp_nb = 0;



        // Calculate the number of entries, so we know the element size

        for (j = 0; j < num_tracks; j++)

            tracks[j].has_cue = 0;

        for (j = 0; j < cues->num_entries - i && entry[j].pts == pts; j++) {

            int tracknum = entry[j].stream_idx;

            av_assert0(tracknum>=0 && tracknum<num_tracks);

            if (tracks[tracknum].has_cue && s->streams[tracknum]->codecpar->codec_type != AVMEDIA_TYPE_SUBTITLE)

                continue;

            tracks[tracknum].has_cue = 1;

            ctp_nb ++;

        }



        cuepoint = start_ebml_master(dyn_cp, MATROSKA_ID_POINTENTRY, MAX_CUEPOINT_SIZE(ctp_nb));

        put_ebml_uint(dyn_cp, MATROSKA_ID_CUETIME, pts);



        // put all the entries from different tracks that have the exact same

        // timestamp into the same CuePoint

        for (j = 0; j < num_tracks; j++)

            tracks[j].has_cue = 0;

        for (j = 0; j < cues->num_entries - i && entry[j].pts == pts; j++) {

            int tracknum = entry[j].stream_idx;

            av_assert0(tracknum>=0 && tracknum<num_tracks);

            if (tracks[tracknum].has_cue && s->streams[tracknum]->codecpar->codec_type != AVMEDIA_TYPE_SUBTITLE)

                continue;

            tracks[tracknum].has_cue = 1;

            track_positions = start_ebml_master(dyn_cp, MATROSKA_ID_CUETRACKPOSITION, MAX_CUETRACKPOS_SIZE);

            put_ebml_uint(dyn_cp, MATROSKA_ID_CUETRACK           , entry[j].tracknum   );

            put_ebml_uint(dyn_cp, MATROSKA_ID_CUECLUSTERPOSITION , entry[j].cluster_pos);

            put_ebml_uint(dyn_cp, MATROSKA_ID_CUERELATIVEPOSITION, entry[j].relative_pos);

            if (entry[j].duration != -1)

                put_ebml_uint(dyn_cp, MATROSKA_ID_CUEDURATION    , entry[j].duration);

            end_ebml_master(dyn_cp, track_positions);

        }

        i += j - 1;

        end_ebml_master(dyn_cp, cuepoint);

    }

    end_ebml_master_crc32(pb, &dyn_cp, mkv, cues_element);



    return currentpos;

}
