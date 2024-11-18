static int webm_dash_manifest_cues(AVFormatContext *s, int64_t init_range)

{

    MatroskaDemuxContext *matroska = s->priv_data;

    EbmlList *seekhead_list = &matroska->seekhead;

    MatroskaSeekhead *seekhead = seekhead_list->elem;

    char *buf;

    int64_t cues_start = -1, cues_end = -1, before_pos, bandwidth;

    int i;

    int end = 0;



    // determine cues start and end positions

    for (i = 0; i < seekhead_list->nb_elem; i++)

        if (seekhead[i].id == MATROSKA_ID_CUES)

            break;



    if (i >= seekhead_list->nb_elem) return -1;



    before_pos = avio_tell(matroska->ctx->pb);

    cues_start = seekhead[i].pos + matroska->segment_start;

    if (avio_seek(matroska->ctx->pb, cues_start, SEEK_SET) == cues_start) {

        // cues_end is computed as cues_start + cues_length + length of the

        // Cues element ID + EBML length of the Cues element. cues_end is

        // inclusive and the above sum is reduced by 1.

        uint64_t cues_length = 0, cues_id = 0, bytes_read = 0;

        bytes_read += ebml_read_num(matroska, matroska->ctx->pb, 4, &cues_id);

        bytes_read += ebml_read_length(matroska, matroska->ctx->pb, &cues_length);

        cues_end = cues_start + cues_length + bytes_read - 1;

    }

    avio_seek(matroska->ctx->pb, before_pos, SEEK_SET);

    if (cues_start == -1 || cues_end == -1) return -1;



    // parse the cues

    matroska_parse_cues(matroska);



    // cues start

    av_dict_set_int(&s->streams[0]->metadata, CUES_START, cues_start, 0);



    // cues end

    av_dict_set_int(&s->streams[0]->metadata, CUES_END, cues_end, 0);



    // if the file has cues at the start, fix up the init range so tht

    // it does not include it

    if (cues_start <= init_range)

        av_dict_set_int(&s->streams[0]->metadata, INITIALIZATION_RANGE, cues_start - 1, 0);



    // bandwidth

    bandwidth = webm_dash_manifest_compute_bandwidth(s, cues_start);

    if (bandwidth < 0) return -1;

    av_dict_set_int(&s->streams[0]->metadata, BANDWIDTH, bandwidth, 0);



    // check if all clusters start with key frames

    av_dict_set_int(&s->streams[0]->metadata, CLUSTER_KEYFRAME, webm_clusters_start_with_keyframe(s), 0);



    // store cue point timestamps as a comma separated list for checking subsegment alignment in

    // the muxer. assumes that each timestamp cannot be more than 20 characters long.

    buf = av_malloc_array(s->streams[0]->nb_index_entries, 20 * sizeof(char));

    if (!buf) return -1;

    strcpy(buf, "");

    for (i = 0; i < s->streams[0]->nb_index_entries; i++) {

        int ret = snprintf(buf + end, 20 * sizeof(char),

                           "%" PRId64, s->streams[0]->index_entries[i].timestamp);

        if (ret <= 0 || (ret == 20 && i ==  s->streams[0]->nb_index_entries - 1)) {

            av_log(s, AV_LOG_ERROR, "timestamp too long.\n");


            return AVERROR_INVALIDDATA;

        }

        end += ret;

        if (i != s->streams[0]->nb_index_entries - 1) {

            strncat(buf, ",", sizeof(char));

            end++;

        }

    }

    av_dict_set(&s->streams[0]->metadata, CUE_TIMESTAMPS, buf, 0);




    return 0;

}