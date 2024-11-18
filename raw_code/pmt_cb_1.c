static void pmt_cb(MpegTSFilter *filter, const uint8_t *section, int section_len)

{

    MpegTSContext *ts = filter->u.section_filter.opaque;

    SectionHeader h1, *h = &h1;

    PESContext *pes;

    AVStream *st;

    const uint8_t *p, *p_end, *desc_list_end, *desc_end;

    int program_info_length, pcr_pid, pid, stream_type;

    int desc_list_len, desc_len, desc_tag;

    int comp_page = 0, anc_page = 0; /* initialize to kill warnings */

    char language[4] = {0}; /* initialize to kill warnings */



#ifdef DEBUG_SI

    av_log(ts->stream, AV_LOG_DEBUG, "PMT: len %i\n", section_len);

    av_hex_dump_log(ts->stream, AV_LOG_DEBUG, (uint8_t *)section, section_len);

#endif

    p_end = section + section_len - 4;

    p = section;

    if (parse_section_header(h, &p, p_end) < 0)

        return;

#ifdef DEBUG_SI

    av_log(ts->stream, AV_LOG_DEBUG, "sid=0x%x sec_num=%d/%d\n",

           h->id, h->sec_num, h->last_sec_num);

#endif

    if (h->tid != PMT_TID)

        return;



    clear_program(ts, h->id);

    pcr_pid = get16(&p, p_end) & 0x1fff;

    if (pcr_pid < 0)

        return;

    add_pid_to_pmt(ts, h->id, pcr_pid);

#ifdef DEBUG_SI

    av_log(ts->stream, AV_LOG_DEBUG, "pcr_pid=0x%x\n", pcr_pid);

#endif

    program_info_length = get16(&p, p_end) & 0xfff;

    if (program_info_length < 0)

        return;

    p += program_info_length;

    if (p >= p_end)

        return;

    for(;;) {

        language[0] = 0;

        st = 0;

        stream_type = get8(&p, p_end);

        if (stream_type < 0)

            break;

        pid = get16(&p, p_end) & 0x1fff;

        if (pid < 0)

            break;

        desc_list_len = get16(&p, p_end) & 0xfff;

        if (desc_list_len < 0)

            break;

        desc_list_end = p + desc_list_len;

        if (desc_list_end > p_end)

            break;

        for(;;) {

            desc_tag = get8(&p, desc_list_end);

            if (desc_tag < 0)

                break;

            if (stream_type == STREAM_TYPE_PRIVATE_DATA) {

                if((desc_tag == 0x6A) || (desc_tag == 0x7A)) {

                    /*assume DVB AC-3 Audio*/

                    stream_type = STREAM_TYPE_AUDIO_AC3;

                } else if(desc_tag == 0x7B) {

                    /* DVB DTS audio */

                    stream_type = STREAM_TYPE_AUDIO_DTS;

                }

            }

            desc_len = get8(&p, desc_list_end);

            desc_end = p + desc_len;

            if (desc_end > desc_list_end)

                break;

#ifdef DEBUG_SI

            av_log(ts->stream, AV_LOG_DEBUG, "tag: 0x%02x len=%d\n",

                   desc_tag, desc_len);

#endif

            switch(desc_tag) {

            case DVB_SUBT_DESCID:

                if (stream_type == STREAM_TYPE_PRIVATE_DATA)

                    stream_type = STREAM_TYPE_SUBTITLE_DVB;



                language[0] = get8(&p, desc_end);

                language[1] = get8(&p, desc_end);

                language[2] = get8(&p, desc_end);

                language[3] = 0;

                get8(&p, desc_end);

                comp_page = get16(&p, desc_end);

                anc_page = get16(&p, desc_end);



                break;

            case 0x0a: /* ISO 639 language descriptor */

                language[0] = get8(&p, desc_end);

                language[1] = get8(&p, desc_end);

                language[2] = get8(&p, desc_end);

                language[3] = 0;

                break;

            default:

                break;

            }

            p = desc_end;

        }

        p = desc_list_end;



#ifdef DEBUG_SI

        av_log(ts->stream, AV_LOG_DEBUG, "stream_type=%d pid=0x%x\n",

               stream_type, pid);

#endif



        /* now create ffmpeg stream */

        switch(stream_type) {

        case STREAM_TYPE_AUDIO_MPEG1:

        case STREAM_TYPE_AUDIO_MPEG2:

        case STREAM_TYPE_VIDEO_MPEG1:

        case STREAM_TYPE_VIDEO_MPEG2:

        case STREAM_TYPE_VIDEO_MPEG4:

        case STREAM_TYPE_VIDEO_H264:

        case STREAM_TYPE_VIDEO_VC1:

        case STREAM_TYPE_AUDIO_AAC:

        case STREAM_TYPE_AUDIO_AC3:

        case STREAM_TYPE_AUDIO_DTS:

        case STREAM_TYPE_SUBTITLE_DVB:

            if(ts->pids[pid]){

                assert(ts->pids[pid]->type == MPEGTS_PES);

                pes= ts->pids[pid]->u.pes_filter.opaque;

                st= pes->st;

            }else{

                pes = add_pes_stream(ts, pid, pcr_pid, stream_type);

                if (pes)

                    st = new_pes_av_stream(pes, 0);

            }

            add_pid_to_pmt(ts, h->id, pid);

            if(st)

                av_program_add_stream_index(ts->stream, h->id, st->index);

            break;

        default:

            /* we ignore the other streams */

            break;

        }



        if (st) {

            if (language[0] != 0) {

                memcpy(st->language, language, 4);

            }



            if (stream_type == STREAM_TYPE_SUBTITLE_DVB) {

                st->codec->sub_id = (anc_page << 16) | comp_page;

            }

        }

    }

    /* all parameters are there */

    ts->stop_parse++;

    mpegts_close_filter(ts, filter);

}
