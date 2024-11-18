static int aiff_read_header(AVFormatContext *s,

                            AVFormatParameters *ap)

{

    int size, filesize;

    offset_t offset = 0;

    uint32_t tag;

    unsigned version = AIFF_C_VERSION1;

    ByteIOContext *pb = s->pb;

    AVStream * st = s->streams[0];



    /* check FORM header */

    filesize = get_tag(pb, &tag);

    if (filesize < 0 || tag != MKTAG('F', 'O', 'R', 'M'))

        return AVERROR_INVALIDDATA;



    /* AIFF data type */

    tag = get_le32(pb);

    if (tag == MKTAG('A', 'I', 'F', 'F'))       /* Got an AIFF file */

        version = AIFF;

    else if (tag != MKTAG('A', 'I', 'F', 'C'))  /* An AIFF-C file then */

        return AVERROR_INVALIDDATA;



    filesize -= 4;



    st = av_new_stream(s, 0);

    if (!st)

        return AVERROR(ENOMEM);



    while (filesize > 0) {

        /* parse different chunks */

        size = get_tag(pb, &tag);

        if (size < 0)

            return size;



        filesize -= size + 8;



        switch (tag) {

        case MKTAG('C', 'O', 'M', 'M'):     /* Common chunk */

            /* Then for the complete header info */

            st->nb_frames = get_aiff_header (pb, st->codec, size, version);

            if (st->nb_frames < 0)

                return st->nb_frames;

            if (offset > 0) // COMM is after SSND

                goto got_sound;

            break;

        case MKTAG('F', 'V', 'E', 'R'):     /* Version chunk */

            version = get_be32(pb);

            break;

        case MKTAG('N', 'A', 'M', 'E'):     /* Sample name chunk */

            get_meta (pb, s->title, sizeof(s->title), size);

            break;

        case MKTAG('A', 'U', 'T', 'H'):     /* Author chunk */

            get_meta (pb, s->author, sizeof(s->author), size);

            break;

        case MKTAG('(', 'c', ')', ' '):     /* Copyright chunk */

            get_meta (pb, s->copyright, sizeof(s->copyright), size);

            break;

        case MKTAG('A', 'N', 'N', 'O'):     /* Annotation chunk */

            get_meta (pb, s->comment, sizeof(s->comment), size);

            break;

        case MKTAG('S', 'S', 'N', 'D'):     /* Sampled sound chunk */

            offset = get_be32(pb);      /* Offset of sound data */

            get_be32(pb);               /* BlockSize... don't care */

            offset += url_ftell(pb);    /* Compute absolute data offset */

            if (st->codec->codec_id)    /* Assume COMM already parsed */

                goto got_sound;

            if (url_is_streamed(pb)) {

                av_log(s, AV_LOG_ERROR, "file is not seekable\n");


            }

            url_fskip(pb, size - 8);

            break;

        case MKTAG('w', 'a', 'v', 'e'):



            st->codec->extradata = av_mallocz(size + FF_INPUT_BUFFER_PADDING_SIZE);

            if (!st->codec->extradata)

                return AVERROR(ENOMEM);

            st->codec->extradata_size = size;

            get_buffer(pb, st->codec->extradata, size);

            break;

        default: /* Jump */

            if (size & 1)   /* Always even aligned */

                size++;

            url_fskip (pb, size);

        }

    }



    /* End of loop and didn't get sound */

    return AVERROR_INVALIDDATA;



got_sound:

    /* Now positioned, get the sound data start and end */

    if (st->nb_frames)

        s->file_size = st->nb_frames * st->codec->block_align;



    av_set_pts_info(st, 64, 1, st->codec->sample_rate);

    st->start_time = 0;

    st->duration = st->codec->frame_size ?

        st->nb_frames * st->codec->frame_size : st->nb_frames;



    /* Position the stream at the first block */

    url_fseek(pb, offset, SEEK_SET);



    return 0;

}