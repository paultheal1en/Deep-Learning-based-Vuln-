static int write_manifest(AVFormatContext *s, int final)

{

    DASHContext *c = s->priv_data;

    AVIOContext *out;

    char temp_filename[1024];

    int ret, i;

    const char *proto = avio_find_protocol_name(s->filename);

    int use_rename = proto && !strcmp(proto, "file");

    static unsigned int warned_non_file = 0;

    AVDictionaryEntry *title = av_dict_get(s->metadata, "title", NULL, 0);



    if (!use_rename && !warned_non_file++)

        av_log(s, AV_LOG_ERROR, "Cannot use rename on non file protocol, this may lead to races and temporary partial files\n");



    snprintf(temp_filename, sizeof(temp_filename), use_rename ? "%s.tmp" : "%s", s->filename);

    ret = s->io_open(s, &out, temp_filename, AVIO_FLAG_WRITE, NULL);

    if (ret < 0) {

        av_log(s, AV_LOG_ERROR, "Unable to open %s for writing\n", temp_filename);

        return ret;

    }

    avio_printf(out, "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n");

    avio_printf(out, "<MPD xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n"

                "\txmlns=\"urn:mpeg:dash:schema:mpd:2011\"\n"

                "\txmlns:xlink=\"http://www.w3.org/1999/xlink\"\n"

                "\txsi:schemaLocation=\"urn:mpeg:DASH:schema:MPD:2011 http://standards.iso.org/ittf/PubliclyAvailableStandards/MPEG-DASH_schema_files/DASH-MPD.xsd\"\n"

                "\tprofiles=\"urn:mpeg:dash:profile:isoff-live:2011\"\n"

                "\ttype=\"%s\"\n", final ? "static" : "dynamic");

    if (final) {

        avio_printf(out, "\tmediaPresentationDuration=\"");

        write_time(out, c->total_duration);

        avio_printf(out, "\"\n");

    } else {

        int64_t update_period = c->last_duration / AV_TIME_BASE;

        char now_str[100];

        if (c->use_template && !c->use_timeline)

            update_period = 500;

        avio_printf(out, "\tminimumUpdatePeriod=\"PT%"PRId64"S\"\n", update_period);

        avio_printf(out, "\tsuggestedPresentationDelay=\"PT%"PRId64"S\"\n", c->last_duration / AV_TIME_BASE);

        if (!c->availability_start_time[0] && s->nb_streams > 0 && c->streams[0].nb_segments > 0) {

            format_date_now(c->availability_start_time, sizeof(c->availability_start_time));

        }

        if (c->availability_start_time[0])

            avio_printf(out, "\tavailabilityStartTime=\"%s\"\n", c->availability_start_time);

        format_date_now(now_str, sizeof(now_str));

        if (now_str[0])

            avio_printf(out, "\tpublishTime=\"%s\"\n", now_str);

        if (c->window_size && c->use_template) {

            avio_printf(out, "\ttimeShiftBufferDepth=\"");

            write_time(out, c->last_duration * c->window_size);

            avio_printf(out, "\"\n");

        }

    }

    avio_printf(out, "\tminBufferTime=\"");

    write_time(out, c->last_duration * 2);

    avio_printf(out, "\">\n");

    avio_printf(out, "\t<ProgramInformation>\n");

    if (title) {

        char *escaped = xmlescape(title->value);

        avio_printf(out, "\t\t<Title>%s</Title>\n", escaped);

        av_free(escaped);

    }

    avio_printf(out, "\t</ProgramInformation>\n");



    if (c->window_size && s->nb_streams > 0 && c->streams[0].nb_segments > 0 && !c->use_template) {

        OutputStream *os = &c->streams[0];

        int start_index = FFMAX(os->nb_segments - c->window_size, 0);

        int64_t start_time = av_rescale_q(os->segments[start_index]->time, s->streams[0]->time_base, AV_TIME_BASE_Q);

        avio_printf(out, "\t<Period id=\"0\" start=\"");

        write_time(out, start_time);

        avio_printf(out, "\">\n");

    } else {

        avio_printf(out, "\t<Period id=\"0\" start=\"PT0.0S\">\n");

    }



    for (i = 0; i < c->nb_as; i++) {

        if ((ret = write_adaptation_set(s, out, i)) < 0)

            return ret;

    }

    avio_printf(out, "\t</Period>\n");



    if (c->utc_timing_url)

        avio_printf(out, "\t<UTCTiming schemeIdUri=\"urn:mpeg:dash:utc:http-xsdate:2014\" value=\"%s\"/>\n", c->utc_timing_url);



    avio_printf(out, "</MPD>\n");

    avio_flush(out);

    ff_format_io_close(s, &out);



    if (use_rename)

        return avpriv_io_move(temp_filename, s->filename);



    return 0;

}
