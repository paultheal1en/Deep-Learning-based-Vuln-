static void write_header(AVFormatContext *s)

{

    double min_buffer_time = 1.0;

    avio_printf(s->pb, "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");

    avio_printf(s->pb, "<MPD\n");

    avio_printf(s->pb, "  xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n");

    avio_printf(s->pb, "  xmlns=\"urn:mpeg:DASH:schema:MPD:2011\"\n");

    avio_printf(s->pb, "  xsi:schemaLocation=\"urn:mpeg:DASH:schema:MPD:2011\"\n");

    avio_printf(s->pb, "  type=\"static\"\n");

    avio_printf(s->pb, "  mediaPresentationDuration=\"PT%gS\"\n",

                get_duration(s));

    avio_printf(s->pb, "  minBufferTime=\"PT%gS\"\n",

                min_buffer_time);

    avio_printf(s->pb, "  profiles=\"urn:webm:dash:profile:webm-on-demand:2012\"");

    avio_printf(s->pb, ">\n");

}
