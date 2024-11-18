static void pat_cb(MpegTSFilter *filter, const uint8_t *section, int section_len)

{

    MpegTSContext *ts = filter->u.section_filter.opaque;

    MpegTSSectionFilter *tssf = &filter->u.section_filter;

    SectionHeader h1, *h = &h1;

    const uint8_t *p, *p_end;

    int sid, pmt_pid;

    AVProgram *program;



    av_log(ts->stream, AV_LOG_TRACE, "PAT:\n");

    hex_dump_debug(ts->stream, section, section_len);



    p_end = section + section_len - 4;

    p     = section;

    if (parse_section_header(h, &p, p_end) < 0)

        return;

    if (h->tid != PAT_TID)

        return;

    if (ts->skip_changes)

        return;



    if (h->version == tssf->last_ver)

        return;

    tssf->last_ver = h->version;

    ts->stream->ts_id = h->id;



    clear_programs(ts);

    for (;;) {

        sid = get16(&p, p_end);

        if (sid < 0)

            break;

        pmt_pid = get16(&p, p_end);

        if (pmt_pid < 0)

            break;

        pmt_pid &= 0x1fff;



        if (pmt_pid == ts->current_pid)

            break;



        av_log(ts->stream, AV_LOG_TRACE, "sid=0x%x pid=0x%x\n", sid, pmt_pid);



        if (sid == 0x0000) {

            /* NIT info */

        } else {

            MpegTSFilter *fil = ts->pids[pmt_pid];

            program = av_new_program(ts->stream, sid);

            if (program) {

                program->program_num = sid;

                program->pmt_pid = pmt_pid;

            }

            if (fil)

                if (   fil->type != MPEGTS_SECTION

                    || fil->pid != pmt_pid

                    || fil->u.section_filter.section_cb != pmt_cb)

                    mpegts_close_filter(ts, ts->pids[pmt_pid]);



            if (!ts->pids[pmt_pid])

                mpegts_open_section_filter(ts, pmt_pid, pmt_cb, ts, 1);

            add_pat_entry(ts, sid);

            add_pid_to_pmt(ts, sid, 0); // add pat pid to program

            add_pid_to_pmt(ts, sid, pmt_pid);

        }

    }



    if (sid < 0) {

        int i,j;

        for (j=0; j<ts->stream->nb_programs; j++) {

            for (i = 0; i < ts->nb_prg; i++)

                if (ts->prg[i].id == ts->stream->programs[j]->id)

                    break;

            if (i==ts->nb_prg && !ts->skip_clear)

                clear_avprogram(ts, ts->stream->programs[j]->id);

        }

    }

}
