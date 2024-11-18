static int r3d_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    R3DContext *r3d = s->priv_data;

    Atom atom;

    int err = 0;



    while (!err) {

        if (read_atom(s, &atom) < 0) {

            err = -1;

            break;

        }

        switch (atom.tag) {

        case MKTAG('R','E','D','V'):

            if (s->streams[0]->discard == AVDISCARD_ALL)

                goto skip;

            if (!(err = r3d_read_redv(s, pkt, &atom)))

                return 0;

            break;

        case MKTAG('R','E','D','A'):

            if (!r3d->audio_channels)

                return -1;

            if (s->streams[1]->discard == AVDISCARD_ALL)

                goto skip;

            if (!(err = r3d_read_reda(s, pkt, &atom)))

                return 0;

            break;

        default:

        skip:

            avio_skip(s->pb, atom.size-8);

        }

    }

    return err;

}
