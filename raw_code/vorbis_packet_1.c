static int vorbis_packet(AVFormatContext *s, int idx)

{

    struct ogg *ogg = s->priv_data;

    struct ogg_stream *os = ogg->streams + idx;

    struct oggvorbis_private *priv = os->private;

    int duration;



    /* first packet handling

       here we parse the duration of each packet in the first page and compare

       the total duration to the page granule to find the encoder delay and

       set the first timestamp */

    if (!os->lastpts) {

        int seg;

        uint8_t *last_pkt = os->buf + os->pstart;

        uint8_t *next_pkt = last_pkt;

        int first_duration = 0;



        avpriv_vorbis_parse_reset(&priv->vp);

        duration = 0;

        for (seg = 0; seg < os->nsegs; seg++) {

            if (os->segments[seg] < 255) {

                int d = avpriv_vorbis_parse_frame(&priv->vp, last_pkt, 1);

                if (d < 0) {

                    duration = os->granule;

                    break;

                }

                if (!duration)

                    first_duration = d;

                duration += d;

                last_pkt = next_pkt + os->segments[seg];

            }

            next_pkt += os->segments[seg];

        }

        os->lastpts = os->lastdts   = os->granule - duration;

        s->streams[idx]->start_time = os->lastpts + first_duration;

        if (s->streams[idx]->duration)

            s->streams[idx]->duration -= s->streams[idx]->start_time;

        s->streams[idx]->cur_dts    = AV_NOPTS_VALUE;

        priv->final_pts             = AV_NOPTS_VALUE;

        avpriv_vorbis_parse_reset(&priv->vp);

    }



    /* parse packet duration */

    if (os->psize > 0) {

        duration = avpriv_vorbis_parse_frame(&priv->vp, os->buf + os->pstart, 1);

        if (duration <= 0) {

            os->pflags |= AV_PKT_FLAG_CORRUPT;

            return 0;

        }

        os->pduration = duration;

    }



    /* final packet handling

       here we save the pts of the first packet in the final page, sum up all

       packet durations in the final page except for the last one, and compare

       to the page granule to find the duration of the final packet */

    if (os->flags & OGG_FLAG_EOS) {

        if (os->lastpts != AV_NOPTS_VALUE) {

            priv->final_pts = os->lastpts;

            priv->final_duration = 0;

        }

        if (os->segp == os->nsegs)

            os->pduration = os->granule - priv->final_pts - priv->final_duration;

        priv->final_duration += os->pduration;

    }



    return 0;

}
