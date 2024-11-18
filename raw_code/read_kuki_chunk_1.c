static int read_kuki_chunk(AVFormatContext *s, int64_t size)

{

    AVIOContext *pb = s->pb;

    AVStream *st      = s->streams[0];



    if (size < 0 || size > INT_MAX - FF_INPUT_BUFFER_PADDING_SIZE)

        return -1;



    if (st->codec->codec_id == AV_CODEC_ID_AAC) {

        /* The magic cookie format for AAC is an mp4 esds atom.

           The lavc AAC decoder requires the data from the codec specific

           description as extradata input. */

        int strt, skip;

        MOVAtom atom;



        strt = avio_tell(pb);

        ff_mov_read_esds(s, pb, atom);

        skip = size - (avio_tell(pb) - strt);

        if (skip < 0 || !st->codec->extradata ||

            st->codec->codec_id != AV_CODEC_ID_AAC) {

            av_log(s, AV_LOG_ERROR, "invalid AAC magic cookie\n");

            return AVERROR_INVALIDDATA;

        }

        avio_skip(pb, skip);

    } else if (st->codec->codec_id == AV_CODEC_ID_ALAC) {

#define ALAC_PREAMBLE 12

#define ALAC_HEADER   36

#define ALAC_NEW_KUKI 24

        uint8_t preamble[12];

        if (size < ALAC_NEW_KUKI) {

            av_log(s, AV_LOG_ERROR, "invalid ALAC magic cookie\n");

            avio_skip(pb, size);

            return AVERROR_INVALIDDATA;

        }

        avio_read(pb, preamble, ALAC_PREAMBLE);



        st->codec->extradata = av_mallocz(ALAC_HEADER + FF_INPUT_BUFFER_PADDING_SIZE);

        if (!st->codec->extradata)

            return AVERROR(ENOMEM);



        /* For the old style cookie, we skip 12 bytes, then read 36 bytes.

         * The new style cookie only contains the last 24 bytes of what was

         * 36 bytes in the old style cookie, so we fabricate the first 12 bytes

         * in that case to maintain compatibility. */

        if (!memcmp(&preamble[4], "frmaalac", 8)) {

            if (size < ALAC_PREAMBLE + ALAC_HEADER) {

                av_log(s, AV_LOG_ERROR, "invalid ALAC magic cookie\n");

                av_freep(&st->codec->extradata);

                return AVERROR_INVALIDDATA;

            }

            avio_read(pb, st->codec->extradata, ALAC_HEADER);

            avio_skip(pb, size - ALAC_PREAMBLE - ALAC_HEADER);

        } else {

            AV_WB32(st->codec->extradata, 36);

            memcpy(&st->codec->extradata[4], "alac", 4);

            AV_WB32(&st->codec->extradata[8], 0);

            memcpy(&st->codec->extradata[12], preamble, 12);

            avio_read(pb, &st->codec->extradata[24], ALAC_NEW_KUKI - 12);

            avio_skip(pb, size - ALAC_NEW_KUKI);

        }

        st->codec->extradata_size = ALAC_HEADER;

    } else {

        st->codec->extradata = av_mallocz(size + FF_INPUT_BUFFER_PADDING_SIZE);

        if (!st->codec->extradata)

            return AVERROR(ENOMEM);

        avio_read(pb, st->codec->extradata, size);

        st->codec->extradata_size = size;

    }



    return 0;

}
