static int mov_read_enda(MOVContext *c, ByteIOContext *pb, MOVAtom atom)

{

    AVStream *st = c->fc->streams[c->fc->nb_streams-1];

    int little_endian = get_be16(pb);



    dprintf(c->fc, "enda %d\n", little_endian);

    if (little_endian == 1) {

        switch (st->codec->codec_id) {

        case CODEC_ID_PCM_S24BE:

            st->codec->codec_id = CODEC_ID_PCM_S24LE;

            break;

        case CODEC_ID_PCM_S32BE:

            st->codec->codec_id = CODEC_ID_PCM_S32LE;

            break;

        case CODEC_ID_PCM_F32BE:

            st->codec->codec_id = CODEC_ID_PCM_F32LE;

            break;

        case CODEC_ID_PCM_F64BE:

            st->codec->codec_id = CODEC_ID_PCM_F64LE;

            break;

        default:

            break;

        }

    }

    return 0;

}
