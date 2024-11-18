static int nsv_parse_NSVf_header(AVFormatContext *s, AVFormatParameters *ap)

{

    NSVContext *nsv = s->priv_data;

    ByteIOContext *pb = &s->pb;

    uint32_t tag, tag1, handler;

    int codec_type, stream_index, frame_period, bit_rate, scale, rate;

    unsigned int file_size, size, nb_frames;

    int64_t duration;

    int strings_size;

    int table_entries;

    int table_entries_used;

    int i, n;

    AVStream *st;

    NSVStream *ast;



    PRINT(("%s()\n", __FUNCTION__));



    nsv->state = NSV_UNSYNC; /* in case we fail */

    

    size = get_le32(pb);

    if (size < 28)


    nsv->NSVf_end = size;



    //s->file_size = (uint32_t)get_le32(pb);

    file_size = (uint32_t)get_le32(pb);

    PRINT(("NSV NSVf chunk_size %ld\n", size));

    PRINT(("NSV NSVf file_size %Ld\n", file_size));



    duration = get_le32(pb); /* in ms */

    nsv->duration = duration * AV_TIME_BASE / 1000; /* convert */

    PRINT(("NSV NSVf duration %Ld ms\n", duration));

    // XXX: store it in AVStreams



    strings_size = get_le32(pb);

    table_entries = get_le32(pb);

    table_entries_used = get_le32(pb);

    PRINT(("NSV NSVf info-strings size: %d, table entries: %d, bis %d\n", 

            strings_size, table_entries, table_entries_used));

    if (url_feof(pb))


    

    PRINT(("NSV got header; filepos %Ld\n", url_ftell(pb)));



    if (strings_size > 0) {

        char *strings; /* last byte will be '\0' to play safe with str*() */

        char *p, *endp;

        char *token, *value;

        char quote;



        p = strings = av_mallocz(strings_size + 1);

        endp = strings + strings_size;

        get_buffer(pb, strings, strings_size);

        while (p < endp) {

            while (*p == ' ')

                p++; /* strip out spaces */

            if (p >= endp-2)

                break;

            token = p;

            p = strchr(p, '=');

            if (!p || p >= endp-2)

                break;

            *p++ = '\0';

            quote = *p++;

            value = p;

            p = strchr(p, quote);

            if (!p || p >= endp)

                break;

            *p++ = '\0';

            PRINT(("NSV NSVf INFO: %s='%s'\n", token, value));

            if (!strcmp(token, "ASPECT")) {

                /* don't care */

            } else if (!strcmp(token, "CREATOR") || !strcmp(token, "Author")) {

                strncpy(s->author, value, 512-1);

            } else if (!strcmp(token, "Copyright")) {

                strncpy(s->copyright, value, 512-1);

            } else if (!strcmp(token, "TITLE") || !strcmp(token, "Title")) {

                strncpy(s->title, value, 512-1);

            }

        }

        av_free(strings);

    }

    if (url_feof(pb))


    

    PRINT(("NSV got infos; filepos %Ld\n", url_ftell(pb)));



    if (table_entries_used > 0) {

        nsv->index_entries = table_entries_used;



        nsv->nsvf_index_data = av_malloc(table_entries * sizeof(uint32_t));

        get_buffer(pb, nsv->nsvf_index_data, table_entries * sizeof(uint32_t));

    }



    PRINT(("NSV got index; filepos %Ld\n", url_ftell(pb)));

    

#ifdef DEBUG_DUMP_INDEX

#define V(v) ((v<0x20 || v > 127)?'.':v)

    /* dump index */

    PRINT(("NSV %d INDEX ENTRIES:\n", table_entries));

    PRINT(("NSV [dataoffset][fileoffset]\n", table_entries));

    for (i = 0; i < table_entries; i++) {

        unsigned char b[8];

        url_fseek(pb, size + nsv->nsvf_index_data[i], SEEK_SET);

        get_buffer(pb, b, 8);

        PRINT(("NSV [0x%08lx][0x%08lx]: %02x %02x %02x %02x %02x %02x %02x %02x"

           "%c%c%c%c%c%c%c%c\n",

           nsv->nsvf_index_data[i], size + nsv->nsvf_index_data[i],

           b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], 

           V(b[0]), V(b[1]), V(b[2]), V(b[3]), V(b[4]), V(b[5]), V(b[6]), V(b[7]) ));

    }

    //url_fseek(pb, size, SEEK_SET); /* go back to end of header */

#undef V

#endif

    

    url_fseek(pb, nsv->base_offset + size, SEEK_SET); /* required for dumbdriving-271.nsv (2 extra bytes) */

    

    if (url_feof(pb))


    nsv->state = NSV_HAS_READ_NSVF;

    return 0;

}